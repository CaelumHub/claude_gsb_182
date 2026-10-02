# -*- coding: utf-8 -*-
"""
表达式即时求值（REPL 式的"随手算一下"）。

设计原则：**不另起求值器**，完整复用项目既有的编译、运行与作用域链路——

  1. 词法 / 语法：直接复用 lexer + parser 的表达式文法（Parser._expression），
     因此运算优先级、列表字面量、下标、调用、短路逻辑等与正式程序完全一致；
  2. 代码生成：复用 CodeGenerator 把表达式 AST 翻译成普通字节码（FunctionCode），
     经过同一套窥孔优化，由同一个 VM 执行，列表 / 字符串 / 函数调用 / && 与 ||
     的短路跳转语义不会出现"第二套行为"；
  3. 作用域：表达式在一个临时帧里执行，帧的局部表按"当前帧局部变量 →
     全局变量 / 用户函数 → 内置函数"的顺序覆盖（与 VM 的 LOAD_VAR 查找规则一致），
     因此能访问当前作用域可见的一切名字；
  4. 调试暂停时求值：临时帧挂在当前调用栈之上执行，求值结束（无论成功或出错）
     后把 VM 的帧栈、finished/error/paused、指令计数、输出等**完整还原**，
     调试会话可以照常继续单步 / 续跑，不会被求值污染。

列表等堆对象按 MiniLang 既有语义以引用方式参与运算，因此调用 push() 等函数
产生的副作用与正常执行一致；求值帧本身的临时变量则随帧丢弃。
"""

from typing import Any, Dict, List, Optional, Tuple

from . import ast_nodes as ast
from . import bytecode as bc
from . import codegen as codegen_mod
from . import diagnostics as diag
from . import lexer as lexer_mod
from . import parser as parser_mod
from . import runtime as rt
from .diagnostics import Diagnostic, DiagnosticBag, PHASE_PARSE, PHASE_SEMANTIC
from .diagnostics import KIND_SYNTAX, KIND_RUNTIME, KIND_LIMIT, SEVERITY_ERROR
from .vm import Frame, VMRuntimeError


# 单次表达式求值允许执行的最大指令数（防止表达式触发死循环 / 无限递归拖垮服务）
EVAL_MAX_STEPS = 1_000_000


class ExpressionCompileError(Exception):
    """表达式未通过词法 / 语法 / 名字检查，携带结构化诊断列表。"""

    def __init__(self, diagnostics: List[Diagnostic]):
        super().__init__(diagnostics[0].message if diagnostics else "表达式不合法")
        self.diagnostics = diagnostics


# ---------------------------------------------------------------------------
# 1) 表达式 -> 字节码（复用 lexer / parser / codegen）
# ---------------------------------------------------------------------------
def compile_expression(expression: str, accessible_names) -> bc.FunctionCode:
    """把单个表达式编译成一段可由 VM 执行的 FunctionCode。

    accessible_names: 当前作用域可见的名字集合（局部 + 全局 + 内置），
    决定代码生成使用 LOAD_VAR 还是 LOAD_GLOBAL。
    失败时抛出 ExpressionCompileError。
    """
    text = (expression or "").strip()
    expr_lines = text.split("\n") if text else []
    bag = DiagnosticBag()

    if not text:
        raise ExpressionCompileError([_simple_error(
            "表达式为空：请输入一个合法的 MiniLang 表达式",
            "例如 1 + 2、len(a)、fib(10) 或 s + \"!\"。", 1, 1)])

    # 1) 词法分析（与正式编译同一个 Lexer）
    tokens, lex_bag = lexer_mod.tokenize(text)
    for d in lex_bag.items:
        if not d.source_line and d.line and expr_lines:
            d.source_line = expr_lines[min(d.line, len(expr_lines)) - 1]
        bag.add(d)
    if bag.has_errors:
        raise ExpressionCompileError(bag.items)

    # 2) 语法分析：只解析一个表达式
    # 先拦截"把语句当表达式"的常见误用，给出比通用语法错误更具体的提示
    first = tokens[0]
    _STATEMENT_KEYWORDS = {
        "var": "变量声明（var x = …）", "func": "函数定义（func f(…) {…}）",
        "if": "条件语句", "while": "while 循环", "for": "for 循环",
        "return": "return 语句", "break": "break 语句",
        "continue": "continue 语句", "print": "print 输出语句",
    }
    if first.type in ("var", "func", "if", "while", "for", "return",
                      "break", "continue", "print"):
        hint = _STATEMENT_KEYWORDS.get(first.text, "语句")
        raise ExpressionCompileError([_simple_error(
            f"这是{hint}，不是一个可以单独求值的表达式",
            f"求值面板只接受表达式（如 1 + 2、f(x)、a[i]）；请把{hint}写在上方源码里运行。",
            first.line, first.column, expr_lines, )])

    parser = parser_mod.Parser(tokens, bag)
    expr = parser._expression()
    if bag.has_errors:
        _enrich(bag.items, expr_lines)
        raise ExpressionCompileError(bag.items)
    if expr is None:
        raise ExpressionCompileError([_simple_error(
            "无法解析为表达式", "检查括号、运算符与字面量是否完整。", 1, 1, expr_lines)])
    if not parser._check(parser_mod.T.EOF):
        tok = parser._cur()
        d = diag.parse_unexpected(tok, ["表达式结束"],
                                  expr_lines[tok.line - 1] if 0 <= tok.line - 1 < len(expr_lines) else "")
        bag.add(d)
        raise ExpressionCompileError(bag.items)

    # 3) 表达式里不允许写赋值（求值面板只读观察当前作用域）
    if isinstance(expr, ast.AssignStmt):
        target = expr.target
        name = getattr(target, "name", None) or "左值"
        raise ExpressionCompileError([_simple_error(
            f"求值面板只接受表达式，不能写赋值语句（{name} {expr.op} …）",
            "去掉赋值运算符；如果想修改变量，请在正式代码 / 调试会话中进行。",
            target.line, getattr(target, "column", 1), expr_lines)])

    # 4) 名字检查（复用语义分析的诊断模板与"你是不是想写"建议）
    _check_names(expr, set(accessible_names), bag, expr_lines)
    if bag.has_errors:
        raise ExpressionCompileError(bag.items)

    # 5) 代码生成：在普通 CodeGenerator 上挂一个临时 FunctionCode，
    #    走的仍是同一条 _expr -> 字节码（含短路跳转）+ _optimize 链路。
    gen = codegen_mod.CodeGenerator()
    fc = bc.FunctionCode("<eval>", 0, [])
    gen.current = fc
    gen._locals = set(accessible_names)
    gen._expr(expr)
    fc.emit(bc.OP_RETURN, None, 1)
    gen._optimize(fc)
    return fc


def _check_names(expr: ast.Expr, accessible: set, bag: DiagnosticBag, expr_lines: List[str]):
    """遍历表达式 AST，对每个标识符做"当前作用域是否可见"的轻量检查。"""
    candidates = sorted(accessible)

    def src_line(line):
        return expr_lines[line - 1] if 0 <= line - 1 < len(expr_lines) else ""

    def walk(node):
        if node is None:
            return
        if isinstance(node, ast.Identifier):
            if node.name not in accessible:
                bag.add(diag.semantic_undefined_name(
                    node.name, candidates, node.line, node.column, src_line(node.line)))
            return
        if isinstance(node, ast.UnaryExpr):
            walk(node.operand)
        elif isinstance(node, ast.BinaryExpr) or isinstance(node, ast.LogicalExpr):
            walk(node.left); walk(node.right)
        elif isinstance(node, ast.CallExpr):
            # 被调用者既可能是标识符也可能是下标/其他表达式，统一走子节点遍历
            walk(node.callee)
            for a in node.args:
                walk(a)
        elif isinstance(node, ast.IndexExpr):
            walk(node.target); walk(node.index)
        elif isinstance(node, ast.ListLiteral):
            for e in node.elements:
                walk(e)

    walk(expr)


def _simple_error(message, fix, line=1, column=1, expr_lines=None):
    return Diagnostic(SEVERITY_ERROR, PHASE_PARSE, KIND_SYNTAX, message,
                      line, column, 1, line, column + 1, fix, None,
                      (expr_lines[line - 1] if expr_lines and 0 <= line - 1 < len(expr_lines) else ""))


def _enrich(items, expr_lines):
    for d in items:
        if not d.source_line and 0 <= d.line - 1 < len(expr_lines):
            d.source_line = expr_lines[d.line - 1]


# ---------------------------------------------------------------------------
# 2) 当前作用域可见名字 / 覆盖层
# ---------------------------------------------------------------------------
def scope_names(vm, frame: Optional[Frame]) -> Dict[str, List[str]]:
    """返回当前作用域可见名字（供前端自动补全 / 展示）。"""
    locals_ = sorted(frame.locals.keys()) if frame is not None else []
    funcs = sorted(n for n, v in vm.globals.items() if isinstance(v, rt.RuntimeFunction))
    gvars = sorted(n for n, v in vm.globals.items()
                   if not isinstance(v, rt.RuntimeFunction))
    builtins = sorted(vm.builtins.keys())
    return {"locals": locals_, "globals": gvars, "functions": funcs,
            "builtins": builtins,
            "all": sorted(set(locals_) | set(gvars) | set(funcs) | set(builtins))}


def _build_overlay(vm, frame: Optional[Frame]) -> Dict[str, Any]:
    """构造求值帧的局部表：当前帧局部 -> 全局（含用户函数）-> 内置。"""
    overlay: Dict[str, Any] = {}
    if frame is not None:
        overlay.update(frame.locals)
    overlay.update(vm.globals)
    overlay.update(vm.builtins)
    # exit() 会直接终止整个 VM，在临时求值里禁止调用
    overlay["exit"] = rt.BuiltinFunction("exit", _blocked_exit, None, min_arity=0)
    return overlay


def _blocked_exit(args):
    raise VMRuntimeError(Diagnostic(
        SEVERITY_ERROR, "runtime", KIND_RUNTIME,
        "表达式求值中不允许调用 exit()：它会终止整个程序",
        1, 1, 4, 1, 5, "从表达式中去掉 exit()；需要退出程序请在正式代码里调用。"))


# ---------------------------------------------------------------------------
# 3) 在（可能正暂停于断点的）VM 上执行求值帧
# ---------------------------------------------------------------------------
def run_on_vm(vm, fc: bc.FunctionCode, expression: str,
              frame: Optional[Frame], max_steps: int = EVAL_MAX_STEPS) -> Dict[str, Any]:
    """在 vm 上压入临时帧执行已编译的表达式字节码。

    live=True 表示 VM 正处在调试暂停中：求值后完整还原 VM 状态；
    live=False 表示 VM 是为本次求值新建、程序已跑完，无需还原。
    """
    live = frame is not None and bool(vm.frames)
    base_depth = len(vm.frames)
    expr_lines = expression.strip().split("\n")
    overlay = _build_overlay(vm, frame)

    # ---- 保存现场（仅 live 场景需要还原） ----
    saved = None
    out_len = len(vm.output)
    if live:
        saved = {
            "finished": vm.finished, "error": vm.error,
            "paused": vm.paused, "pause_reason": vm.pause_reason,
            "return_value": vm.return_value,
            "instruction_count": vm.instruction_count,
            "cur_line": vm._cur_line,
        }

    eval_frame = Frame("<eval>", fc)
    eval_frame.locals = overlay
    # standalone 场景下 VM 刚跑完整个程序（finished=True），需复位才能继续驱动求值帧
    if not live:
        vm.finished = False
        vm.error = None
    vm.frames.append(eval_frame)

    error_diag: Optional[Diagnostic] = None
    limit_hit = False
    in_eval_frame = False
    steps = 0
    try:
        while len(vm.frames) > base_depth and vm.error is None:
            if steps >= max_steps:
                limit_hit = True
                break
            vm.step_instruction()
            steps += 1
    finally:
        prints = list(vm.output[out_len:])
        value = None
        if vm.error is None and not limit_hit:
            # 成功：
            #  * live：_do_return 已把结果压回暂停帧操作数栈，取出并还原其栈；
            #  * standalone：临时帧是唯一帧，结果在 vm.return_value。
            if live:
                if frame is not None and frame.stack:
                    value = frame.stack.pop()
            else:
                value = vm.return_value

        if limit_hit:
            cur_line = vm.frames[-1].current_line if vm.frames else 1
            error_diag = Diagnostic(
                SEVERITY_ERROR, "runtime", KIND_LIMIT,
                f"表达式求值执行超过 {max_steps} 条指令，疑似死循环或无限递归，已中止",
                cur_line, 1, 1, cur_line, 2,
                "检查表达式中的递归终止条件与循环函数调用。")
        elif vm.error is not None:
            error_diag = vm.error
            in_eval_frame = (len(vm.frames) > base_depth
                             and vm.frames[-1].code is fc)

        if live:
            # ---- 还原现场：丢弃求值残留帧与全部 VM 级状态改动 ----
            # 兜底清理：理论上 RETURN 后不会再触发步数上限，这里仍确保暂停帧栈干净
            if limit_hit and frame is not None and len(vm.frames) == base_depth and frame.stack:
                frame.stack.pop()
            del vm.frames[base_depth:]
            vm.finished = saved["finished"]
            vm.error = saved["error"]
            vm.paused = saved["paused"]
            vm.pause_reason = saved["pause_reason"]
            vm.return_value = saved["return_value"]
            vm.instruction_count = saved["instruction_count"]
            vm._cur_line = saved["cur_line"]
            del vm.output[out_len:]

    names = scope_names(vm, frame)

    if error_diag is not None:
        d = error_diag.to_dict()
        # 出错位置在求值帧自身的代码里 -> 取表达式原文；
        # 若发生在表达式调用的用户函数内，则取正式程序源码行。
        if in_eval_frame and 0 <= error_diag.line - 1 < len(expr_lines):
            d["source_line"] = expr_lines[error_diag.line - 1]
        else:
            d["source_line"] = vm._line(error_diag.line)
        return {"ok": False, "error": d, "prints": prints,
                "instruction_count": steps, "names": names}

    return {
        "ok": True,
        "value": rt.serialize_value(value),
        "display": _display(value),
        "type": rt.type_name(value),
        "prints": prints,
        "instruction_count": steps,
        "names": names,
    }


def _display(v) -> str:
    """MiniLang 视角的值文本（复用 VM 的打印格式，保持与 print 一致）。"""
    from .vm import _to_display
    return _to_display(v)


# ---------------------------------------------------------------------------
# 4) 一站式入口：编译表达式 + 在指定 VM / 当前帧上求值
# ---------------------------------------------------------------------------
def evaluate(vm, expression: str, frame: Optional[Frame] = None) -> Dict[str, Any]:
    """编译并求值一个表达式。frame 为 None 时只访问全局作用域。

    返回值永远是可直接 JSON 化的 dict；编译类错误携带 diagnostics。
    """
    names = scope_names(vm, frame)
    try:
        fc = compile_expression(expression, names["all"])
    except ExpressionCompileError as e:
        return {"ok": False, "stage": "expression",
                "diagnostics": [d.to_dict() for d in e.diagnostics],
                "names": names}
    return run_on_vm(vm, fc, expression, frame)
