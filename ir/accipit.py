from __future__ import annotations
from lark import Lark, Transformer, ast_utils, Token, UnexpectedInput
from lark.ast_utils import Ast
from dataclasses import dataclass
from typing import Union, Any
import random

import sys
import argparse

sys.setrecursionlimit(50000)  # 加深调用栈深度，因为python没有尾递归优化


class SemanticError(Exception):
    def __init__(self, message):
        super().__init__(message)


STEP = 0
DEBUG = False


class IRNode():
    wrap = True
    return_msg = ""

    def method_wrapper(self, func, step: int):
        def wrapper(*args, **kwargs):
            name = self.__str__().split('\n')[0].strip().rstrip("{").strip()
            prefix = f"STEP {step}.".ljust(10)
            print(f"{prefix} \033[1;32mEval\033[0m   {name} with args={args}")
            result = func(*args, **kwargs)
            msg = self.return_msg if self.return_msg else f"\033[1;33mReturn {result}\033[0m for {name}"
            print(f"{prefix} {msg}")
            return result
        return wrapper

    def __getattribute__(self, name):
        obj = super().__getattribute__(name)
        if name == "eval" and self.wrap:
            global STEP
            step = STEP
            STEP = STEP + 1
            if DEBUG and callable(obj):
                return self.method_wrapper(obj, step)
        return obj

    def __str__(self):
        return self.__class__.__name__ + " " + " ".join(f"({attr})" for attr in self.__dict__.values()) + " "


@dataclass
class IntConst(Ast):
    value: int

    def __str__(self):
        return str(self.value)

    def eval(self) -> int:
        return self.value


@dataclass
class NoneConst():
    def eval(self):
        return 1


@dataclass
class UnitConst():
    def eval(self) -> UnitConst:
        return self


@dataclass
class Ident(IRNode):
    name: str

    def __str__(self):
        return self.name

    def eval(self) -> Any:
        return env.get(self)


Value = Union[IntConst, NoneConst, UnitConst, Ident]


class I32(IRNode):
    def __str__(self):
        return "i32"


class Unit(IRNode):
    def __str__(self):
        return "()"


@dataclass
class Pointer(IRNode):
    name: str


@dataclass
class FunType(IRNode):
    params: list[Token]
    ret: Token

    def __str__(self):
        return f"fn ({', '.join(str(param) for param in self.params)}) -> {self.ret}"


Type = Union[I32, Unit, Pointer, FunType]


@dataclass
class Ptr(IRNode):
    addr: int


class Environment():
    def __init__(self):
        self.global_env: dict[str, Any] = {}
        self.frames: list[dict[str, Any]] = []
        self.memory: list[int] = [
            random.randint(1, 0xffff) for _ in range(1024)]
        self.capacity: int = 1024
        self.size: int = 0

    def push_frame(self):
        self.frames.append({})

    def pop_frame(self):
        self.frames.pop()

    def allocate(self, size: int, init: list[int] = []) -> Ptr:
        if self.size + size > self.capacity:
            size_lacked = self.size + size - self.capacity
            size_to_extend = (size_lacked + 1023) // 1024 * 1024
            self.memory.extend([
                random.randint(1, 0xffff) for _ in range(1024)])
            self.capacity += size_to_extend
        addr = self.size
        self.size += size
        if init:
            self.memory[addr:addr+size] = init
        return Ptr(addr)

    def add_global(self, name, value: Any):
        name = name.__str__()
        if name in self.global_env:
            raise SemanticError(f"Global identifier {name} is defined twice.")
        self.global_env[name] = value

    def get_global(self, name) -> Any:
        return self.global_env.get(name.__str__())

    def add_local(self, name, value: Any):
        name = name.__str__()
        self.frames[-1][name] = value

    def get_local(self, name) -> Any:
        return self.frames[-1].get(name.__str__())

    def get(self, name: Ident) -> Any:
        if name.name.startswith("@"):
            return self.get_global(name)
        else:
            return self.get_local(name)

    def store(self, name: Ident, value: int):
        ptr = self.get(name)
        if ptr is None:
            raise SemanticError(f"{name} is not defined.")
        if not isinstance(ptr, Ptr):
            raise SemanticError(f"{name} is not a pointer.")
        self.memory[ptr.addr] = value

    def load(self, name: Ident) -> int:
        ptr = self.get(name)
        if ptr is None:
            raise SemanticError(f"{name} is not defined.")
        if not isinstance(ptr, Ptr):
            raise SemanticError(f"{name} is not a pointer.")
        return self.memory[ptr.addr]

    def clear(self):
        self.global_env.clear()
        self.frames.clear()


env = Environment()


@dataclass
class BinExpr(IRNode, Ast):
    binop: Token
    v1: IntConst | Ident
    v2: IntConst | Ident

    def eval(self):
        v1 = self.v1.eval()
        v2 = self.v2.eval()
        if self.binop == "add":
            return v1 + v2
        elif self.binop == "sub":
            return v1 - v2
        elif self.binop == "mul":
            return v1 * v2
        elif self.binop == "div":
            return int(v1 / v2)
        elif self.binop == "rem":
            return v1 % v2
        elif self.binop == "and":
            return v1 & v2
        elif self.binop == "or":
            return v1 | v2
        elif self.binop == "xor":
            return v1 ^ v2
        elif self.binop == "eq":
            return v1 == v2
        elif self.binop == "ne":
            return v1 != v2
        elif self.binop == "lt":
            return v1 < v2
        elif self.binop == "le":
            return v1 <= v2
        elif self.binop == "gt":
            return v1 > v2
        elif self.binop == "ge":
            return v1 >= v2
        else:
            raise SemanticError(f"Unknown binop {self.binop}")


@dataclass
class Alloca(IRNode, Ast):
    tpe: Type
    size: IntConst

    def eval(self) -> Ptr:
        return env.allocate(self.size.eval())


@dataclass
class Load(IRNode, Ast):
    name: Ident

    def eval(self):
        return env.load(self.name)


@dataclass
class Store(IRNode, Ast):
    wrap = False
    value: Ident
    name: Ident

    def eval(self):
        env.store(self.name, self.value.eval())


@dataclass
class Gep(IRNode):
    tpe: Type
    name: Ident
    offsets: list[tuple[IntConst, Union[IntConst, NoneConst]]]

    def __str__(self):
        indexing = ", ".join(f"{idx} < {dim}" for idx, dim in self.offsets)
        return f"offset {self.tpe}, {self.name}, {indexing}"

    def eval(self) -> Ptr:
        addr = env.get(self.name).addr
        p_addr = 0
        for idx, dim in self.offsets:
            p_addr = p_addr * dim.eval() + idx.eval()
        return Ptr(addr + p_addr)


@dataclass
class Fncall(IRNode):
    name: Ident
    args: list[Value]

    def __str__(self):
        return f"call {self.name} ({', '.join(str(arg) for arg in self.args)})"

    def eval(self):
        if self.name.__str__() == "@write":
            print(self.args[0].eval())
            return 0
        elif self.name.__str__() == "@read":
            return int(input())
        fun = env.get_global(self.name)
        if not isinstance(fun, FunDefn) and not isinstance(fun, FunDecl):
            raise SemanticError(f"{self.name} is not a function.")
        return fun.eval(self.args)


ValueBindingOp = Union[BinExpr, Gep, Fncall, Alloca, Load, Store]


@dataclass
class ValueBinding(IRNode):
    name: Ident
    op: ValueBindingOp

    def __str__(self):
        return f"\tlet {self.name} = {self.op}"

    def eval(self):
        value = self.op.eval()
        env.add_local(self.name, value)
        self.return_msg = f"\033[1;33mBind  \033[0m {value} to {self.name}"


@dataclass
class Br(IRNode, Ast):
    cond: Value
    label1: Ident
    label2: Ident

    def eval(self) -> int:
        target = self.label1 if self.cond.eval() else self.label2
        return env.get(target).eval()


@dataclass
class Jmp(IRNode, Ast):
    label: Ident

    def eval(self) -> int:
        return env.get(self.label).eval()


@dataclass
class Ret(IRNode):
    value: IntConst | Ident | UnitConst

    def eval(self) -> Value:
        return self.value.eval()


Terminator = Union[Br, Jmp, Ret]


@dataclass
class PList(IRNode):
    wrap = False
    params: list[tuple[Ident, Type]]

    def __str__(self):
        return "(" + ", ".join(f"{name}: {tpe}" for name, tpe in self.params) + ")"

    def eval(self, values: list[Ident | IntConst]):
        values = [value.eval() for value in values]
        env.push_frame()
        for (name, _), value in zip(self.params, values):
            env.add_local(name, value)


@dataclass
class BasicBlock(IRNode):
    wrap = False
    label: Ident
    insts: list[ValueBinding | Terminator]

    def __str__(self):
        return f"{self.label}:\n" + "\n".join(str(binding) for binding in self.bindings) + f"\n{self.terminator}"

    def eval(self) -> int | UnitConst:
        for idx, inst in enumerate(self.insts):
            if isinstance(inst, ValueBinding):
                inst.eval()
            else:
                if idx != len(self.insts) - 1:
                    raise SemanticError(
                        f"Terminator instruction {inst} is not the last instruction in block {self.label}.")
                return inst.eval()


@dataclass
class Body(IRNode):
    wrap = False
    bbs: list[BasicBlock]

    def __str__(self):
        return "{\n" + "\n".join(str(bb) for bb in self.bbs) + "\n}"

    def eval(self) -> int:
        return self.bbs[0].eval()


class GlobalDecl:
    name: Ident
    tpe: Type
    size: IntConst
    values: list[IntConst]

    def __init__(self, name: Ident, tpe: Type, size: IntConst, values: list[IntConst]):
        self.name = name
        self.tpe = tpe
        self.size = size
        self.values = values
        if values and len(values) != size.eval():
            raise SemanticError(
                f"Global array {name} has size {size} but {len(values)} values are provided.")
        ptr = env.allocate(size.eval(), [value.eval() for value in values])
        env.add_global(name, ptr)

    def __str__(self):
        if self.values:
            return f"{self.name} : {self.tpe}, {self.size} = [{', '.join(str(value) for value in self.values)}]"
        else:
            return f"{self.name} : {self.tpe}, {self.size}"


class FunDefn(IRNode, Ast):
    name: Ident
    params: PList
    ret: Type
    body: Body

    def __init__(self, name: Ident, params: PList, ret: Type, body: Body):
        self.name = name
        self.params = params
        self.ret = ret
        self.body = body
        env.add_global(name, self)

    def __str__(self):
        return f"fn {self.name} ({self.params}) -> {self.ret} {self.body}"

    def eval(self, args: list[Value]) -> int:
        self.params.eval(args)
        for bb in self.body.bbs:
            env.add_local(bb.label, bb)
        return_value = self.body.eval()
        env.pop_frame()
        return return_value


class FunDecl(IRNode, Ast):
    name: Ident
    params: PList
    ret: Type

    def __init__(self, name: Ident, params: PList, ret: Type):
        self.name = name
        self.params = params
        self.ret = ret
        env.add_global(name, self)

    def __str__(self):
        return f"fn {self.name} ({self.params}) -> {self.ret};"


Decl = Union[GlobalDecl, FunDefn, FunDecl]


@dataclass
class Program():
    decls: list[Decl]

    def __str__(self):
        return "\n".join(str(decl) for decl in self.decls)


class BaseTransformer(Transformer):
    def name(_, children): return "".join(children)
    def i32(_, _token): return I32()
    def unit(_, _token): return Unit()
    def int_const(_, n): return IntConst(int(n[0]))
    def none_const(_, _token): return NoneConst()
    def unit_const(_, _token): return UnitConst()
    def SIGNED_INT(_, n): return int(n)
    def function_type(_, items): return FunType(items[:-1], items[-1])
    def pointer(_, items): return Pointer(items[0].__str__() + "*")
    def ident(_, items): return Ident(items[0] + items[1])
    def gep(_, items): return Gep(items[0], items[1], [
        (items[i], items[i+1]) for i in range(2, len(items), 2)])

    def fncall(_, items): return Fncall(items[0], items[1:])
    def value_binding_untyped(
        _, items): return ValueBinding(items[0], items[1])

    def value_binding_typed(_, items): return ValueBinding(items[0], items[2])
    def ret(_, items): return Ret(items[1])
    def plist(_, items): return PList(
        [(items[i], items[i+1]) for i in range(0, len(items), 2)])

    def bb(_, items): return BasicBlock(items[0], items[1:])
    def body(_, items): return Body(items)
    def global_decl(_, items): return GlobalDecl(
        items[0], items[1], items[2], items[3:])

    def program(_, items): return Program(items)


accipit_grammar = """
    ?start : program

    name : /[a-zA-Z.-_]/ /[a-zA-Z0-9.-_]/*
    ident : /@/ (name | INT)
    | /#/ (name | INT)
    | /%/ (name | INT) 

    int_const : SIGNED_INT -> int_const
    none_const : "none"
    unit_const : "()"
    ?const : int_const | none_const | unit_const

    ?value : ident | const
    
    type : /i32/ -> i32
    | "()" -> unit
    | type "*" -> pointer
    | "fn" "(" (type ("," type)*)? ")" "->"  type -> function_type
    
    value_binding_untyped : "let" ident "=" (bin_expr | gep | fncall | alloca | load | store)
    value_binding_typed : "let" ident ":" type "=" (bin_expr | gep | fncall | alloca | load | store)
    ?value_binding : value_binding_untyped | value_binding_typed
    ?terminator : br | jmp | ret
    
    ?binop : /add/ | /sub/ | /mul/ | /div/ | /rem/ | /and/ | /or/ | /xor/ | /eq/ | /ne/ | /lt/ | /le/ | /gt/ | /ge/
    bin_expr : binop value "," value
    alloca : "alloca" type "," int_const
    load : "load" ident
    store : "store" value "," ident 
    gep : "offset" type "," ident ( "," "[" value "<" (int_const | none_const) "]" )+
    fncall : "call" ident ("," value)*
    
    br : "br" value "," "label" ident "," "label" ident
    jmp : "jmp" "label" ident
    ret : /(?<!\w)ret(?!\w)/ value
    
    ?plist : (ident ":" type ("," ident ":" type)*)?
    ?label : ident ":"
    ?bb : label (value_binding| terminator)+
    body : "{" bb* "}"
    
    global_decl : ident ":" "region" type "," int_const ("=" "[" value ("," value)* "]")?
    fun_defn : "fn" ident "(" plist ")" "->" type body
    fun_decl : "fn" ident "(" plist ")" "->" type ";"
    program : (global_decl | fun_defn | fun_decl)*

    %import common.WS
    %import common.CPP_COMMENT
    %import common.C_COMMENT
    %import common.INT
    %import common.SIGNED_INT

    %ignore WS
    %ignore CPP_COMMENT
    %ignore C_COMMENT
"""

this_module = sys.modules[__name__]
accipit_transformer = ast_utils.create_transformer(
    this_module, BaseTransformer())
parser = Lark(accipit_grammar)


def parse(file: str) -> Program:
    with open(file) as f:
        text = f.read()
    try:
        ast = parser.parse(text)
        result: Program = accipit_transformer.transform(ast)
        return result
    except UnexpectedInput as e:
        print(e.get_context(text))
        print(f"Syntax error at position {e.column}: {e}")
        exit(1)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Interpreter for Accipit IR")
    arg_parser.add_argument("file", type=str, help="the IR file to interpret.")
    arg_parser.add_argument(
        "-d", "--debug", action="store_true", help="whether to print debug info.")
    args = arg_parser.parse_args()
    program = parse(args.file)
    if args.debug:
        DEBUG = True
        print(f"The parsed AST is:\n{program}")
        print("\n-----------------The evaluation starts here-----------------\n")
    main = env.global_env.get("@main")
    if main is None or not isinstance(main, FunDefn):
        raise SemanticError("Main function is not defined.")
    return_value = main.eval([])
    colored_return_value = f"\033[1;32m{return_value}\033[0m" if return_value == 0 else f"\033[1;31m{return_value}\033[0m"
    print(f'Exit with code {colored_return_value} within {STEP} steps.')
    exit(return_value)
