# -*- coding: utf-8 -*-
"""
调试表达式求值。

表达式面板不另建解释器：仍然经过词法/语法分析和现有 CodeGenerator 生成字节码，
再把一个临时求值帧压到暂停中的同一个 VM 上执行。临时帧复制当前帧的局部变量，
名字回退到 VM 全局环境，因此当前局部变量、用户函数和内置函数都沿用同一条作用域
与运行时链路；列表、字符串、短路逻辑和函数调用也由 VM 的既有指令实现。
"""

from . import ast_nodes as ast
from . import bytecode as bc
from . import codegen as codegen_mod
from . import diagnostics as diag
from . import lexer as lexer_mod
from . import parser as parser_mod
from . import runtime as rt
from . import vm as vm_mod


class ExpressionResult:
    def __init__(self, ok=False, value=None, diagnostics=None, output=None):
        self.ok = ok
        self.value = value
        self.diagnostics = diagnostics if diagnostics is not None else diag.DiagnosticBag()
        self.output = output if output is not None else []

    def to_dict(self):
        return {
            "ok": self.ok,
            "value": rt.serialize_value(self.value) if self.ok else None,
            "diagnostics": self.diagnostics.to_list(),
            "output": list(self.output),
        }


def compile_expression(source, local_names=None):
    """把一个表达式编译成独立 FunctionCode（复用 Lexer/Parser/CodeGenerator）。"""
    text = (source or "").strip()
    diagnostics = diag.DiagnosticBag()
    if not text:
        diagnostics.error(
            "表达式不能为空", phase=diag.PHASE_PARSE, kind=diag.KIND_SYNTAX,
            line=1, column=1, length=1, fix="输入一个表达式，例如 n + 1、len(a) 或 a[0]。",
            source_line=text)
        return None, diagnostics

    tokens, lex_diags = lexer_mod.tokenize(text)
    diagnostics.items.extend(lex_diags.items)
    _enrich_expression_lines(diagnostics, text)
    if lex_diags.has_errors:
        return None, diagnostics

    expr = parser_mod.parse_expression(tokens, diagnostics)
    _enrich_expression_lines(diagnostics, text)
    if diagnostics.has_errors or expr is None:
        return None, diagnostics

    if isinstance(expr, ast.AssignStmt):
        diagnostics.error(
            "表达式面板不能执行赋值操作", phase=diag.PHASE_PARSE, kind=diag.KIND_SYNTAX,
            line=expr.line, column=expr.column, length=1,
            fix="只输入要查看的表达式；需要修改变量时继续单步执行原程序。",
            source_line=text)
        return None, diagnostics

    generator = codegen_mod.CodeGenerator()
    code = bc.FunctionCode("<eval>", 0, [])
    generator.current = code
    generator._locals = set(local_names or ())
    generator._expr(expr)
    code.emit(bc.OP_RETURN, None, 1)
    for i, ins in enumerate(code.instructions):
        ins.offset = i
    code.line_to_offset = {1: 0}
    return code, diagnostics


def evaluate(vm, expression):
    """在暂停的 VM 当前作用域中执行已编译表达式。"""
    if not vm.frames or vm.finished or vm.error:
        diagnostics = diag.DiagnosticBag()
        diagnostics.error(
            "程序当前未暂停，不能求值表达式", phase=diag.PHASE_RUNTIME,
            kind=diag.KIND_RUNTIME, line=1, column=1, length=1,
            fix="先启动调试并在断点或单步暂停后再求值。", source_line=expression)
        return ExpressionResult(False, None, diagnostics)

    current = vm.frames[-1]
    code, diagnostics = compile_expression(expression, current.locals.keys())
    if diagnostics.has_errors:
        return ExpressionResult(False, None, diagnostics)
    frame = vm_mod.Frame("<eval>", code)
    frame.locals = dict(current.locals)

    saved_frames = list(vm.frames)
    saved_finished = vm.finished
    saved_paused = vm.paused
    saved_pause_reason = vm.pause_reason
    saved_error = vm.error
    saved_return_value = vm.return_value
    saved_output_len = len(vm.output)
    saved_instruction_count = vm.instruction_count
    saved_profiler = vm.profiler
    saved_debugger = vm.debugger
    saved_caller_len = len(current.stack)
    result = None

    vm.profiler = None
    vm.debugger = None
    vm.error = None
    vm.finished = False
    vm.paused = False
    vm.pause_reason = None
    vm.frames.append(frame)

    def stop_after_eval(engine):
        return engine.frames is not None and len(engine.frames) <= len(saved_frames)

    try:
        vm.run(stop_after_eval, None)
        if vm.error:
            raise vm_mod.VMRuntimeError(vm.error)
        if vm.finished and not saved_finished:
            diagnostics.error(
                "表达式调用 exit()，已阻止调试会话退出", phase=diag.PHASE_RUNTIME,
                kind=diag.KIND_RUNTIME, line=1, column=1, length=4,
                fix="从表达式中移除 exit() 调用。", source_line=expression)
        elif len(current.stack) > saved_caller_len:
            result = current.stack[-1]
        else:
            diagnostics.error(
                "表达式没有产生值", phase=diag.PHASE_RUNTIME, kind=diag.KIND_RUNTIME,
                line=1, column=1, length=1, fix="检查表达式是否可计算。", source_line=expression)
    except vm_mod.VMRuntimeError as e:
        _ensure_expression_source(e.diagnostic, expression)
        diagnostics.add(e.diagnostic)
    except vm_mod.SystemExitSignal:
        diagnostics.error(
            "表达式调用 exit()，已阻止调试会话退出", phase=diag.PHASE_RUNTIME,
            kind=diag.KIND_RUNTIME, line=1, column=1, length=4,
            fix="从表达式中移除 exit() 调用。", source_line=expression)
    except Exception as e:
        diagnostics.error(
            f"表达式求值失败：{e}", phase=diag.PHASE_RUNTIME, kind=diag.KIND_RUNTIME,
            line=1, column=1, length=1, source_line=expression)
    finally:
        captured_output = list(vm.output[saved_output_len:])
        del vm.output[saved_output_len:]
        vm.frames = saved_frames
        vm.finished = saved_finished
        vm.paused = saved_paused
        vm.pause_reason = saved_pause_reason
        vm.error = saved_error
        vm.return_value = saved_return_value
        vm.instruction_count = saved_instruction_count
        vm.profiler = saved_profiler
        vm.debugger = saved_debugger
        if len(current.stack) > saved_caller_len:
            current.stack.pop()

    ok = not diagnostics.has_errors
    return ExpressionResult(ok, result if ok else None, diagnostics, captured_output)


def _enrich_expression_lines(bag, source):
    for d in bag.items:
        if not d.source_line:
            d.source_line = source


def _ensure_expression_source(diagnostic, expression):
    """直接由表达式字节码报出的错误把定位源文改为表达式；进入用户函数后保留原源码。"""
    if diagnostic.line == 1 and not diagnostic.source_line:
        diagnostic.source_line = expression
