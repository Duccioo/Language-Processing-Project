from lark import Lark, Transformer

# Define the grammar for function definition and function call
grammar = r"""
    start: function_def function_call

    function_def: "function" NAME "(" params ")" "{" "return" expr ";" "}" 
    function_call: NAME "(" args ")" [";"] 

    params: NAME ("," NAME)*
    args: NUMBER ("," NUMBER)*

    expr: NAME op expr | NAME

    op: "+" | "*"

    NAME: /[a-zA-Z_]\w*/
    NUMBER: /[0-9]+/
    # %ignore " "
    %import common.WS
    %ignore WS
"""


# Define a transformer to process the parsed tree
class FunctionTransformer(Transformer):
    def function_def(self, items):
        name, params, expr = items
        print("////////", items)
        return name, params, expr

    def function_call(self, items):
        name, args = items
        return name, args

    def expr(self, items):
        if len(items) == 1:
            return items[0]
        else:
            return (items[1], items[0], items[2])


# Parse function definition and call
def parse_input(input_str):
    parser = Lark(grammar, parser="lalr", transformer=FunctionTransformer())
    return parser.parse(input_str)


# Execute function call based on parsed information
def execute_function(parsed_info, function_dict):
    name, params, expr = parsed_info
    param_dict = {p: int(val) for p, val in zip(params, function_dict[name][0])}
    expression = expr
    while isinstance(expression, tuple):
        op, param1, param2 = expression
        param1_val = param_dict[param1]
        param2_val = param_dict[param2]
        if op == "+":
            result = param1_val + param2_val
        elif op == "*":
            result = param1_val * param2_val
        param_dict[expression] = result
        expression = expression[2]
    return param_dict[expr]


# Example input string
input_string = """
    function add(a, b) {
        return a + b;
    }

    add(1, 2)
"""

# Define function dictionary to store function definitions and results
function_dict = {}

# Parse input string and store function definition in function_dict
parsed = parse_input(input_string)
print(parsed)
name, params, expr = parsed
function_dict[name] = (params, expr)

# Execute function call and print result
result = execute_function(parsed[1:], function_dict)
print(result)
