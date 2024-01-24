from lark import Lark, Transformer, v_args

# Define the grammar
grammar = r"""
    start: function_definition function_call

    function_definition: "function" NAME "(" parameters ")" "{" "return" expression ";}"
    parameters: (NAME ",")* NAME
    expression: NAME op NAME
    op: "+" | "*"

    function_call: NAME "(" arguments ")"
    arguments: (NUMBER ",")* NUMBER

    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""


# Define a transformer to process the parsed tree
class FunctionTransformer(Transformer):
    @v_args(inline=True)
    def function_definition(self, name, params, expression):
        return {"name": name, "params": params, "expression": expression}

    @v_args(inline=True)
    def function_call(self, name, arguments):
        return {"name": name, "arguments": arguments}

    @v_args(inline=True)
    def parameters(self, *params):
        return params

    @v_args(inline=True)
    def arguments(self, *args):
        return args

    @v_args(inline=True)
    def expression(self, left, op, right):
        return {"left": left, "op": op, "right": right}


# Parse and transform the input
def parse_input(input_string):
    parser = Lark(grammar, parser="lalr", transformer=FunctionTransformer())
    return parser.parse(input_string)


# Execute the function call
def execute_function_call(definition, call_args):
    params = {param: arg for param, arg in zip(definition["params"], call_args)}
    if definition["expression"]["op"] == "+":
        result = sum(params.values())
    elif definition["expression"]["op"] == "*":
        result = 1
        for value in params.values():
            result *= value
    else:
        raise ValueError("Invalid operation")

    return result


# Example input string
input_string = """
function add(x, y) { return x + y;}

add(2, 3)
"""

# Parse the input and execute the function call
parsed_tree = parse_input(input_string)
function_definition = parsed_tree.children[0]
function_call = parsed_tree.children[1]

result = execute_function_call(function_definition, function_call["arguments"])
print(result)
