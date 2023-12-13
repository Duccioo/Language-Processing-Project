from lark import Lark, Transformer, v_args, tree


# Define a transformer to process the parsed tree
# @v_args(inline=True)
class FunctionTransformer(Transformer):
    NAME = str
    NUMBER = float
    func = {
        "name": None,
        "args": [],
        "expression": {"variable": [], "operations": []},
    }

    def function_def(self, args):
        name, parameters, expression = args

        self.func["name"] = name

        if isinstance(parameters, tree.Tree):
            parameters = parameters.children

        for element in parameters:
            self.func["args"].append(element)

        if not set(self.func["expression"]["variable"]) <= set(self.func["args"]):
            print("error: some variables defined in function are not arguments")
            print(self.func["expression"]["variable"], self.func["args"])

        self.func["expression"]["operations"] = expression
        return self.func["name"]

    def function_call(self, args):
        name, arguments = args
        variables_dict = {}

        if isinstance(arguments, tree.Tree):
            arguments = arguments.children

        for i, arg in enumerate(arguments):
            variables_dict[self.func["args"][i]] = arg
        if name != self.func["name"]:
            print("function not found")
        else:
            result = calc_expression(
                variables_dict, self.func["expression"]["operations"]
            )
            return result

    def sum(self, args):
        if isinstance(args[0], float) and isinstance(args[1], float):
            return args[0] + args[1]

        if isinstance(args[0], float) == False and isinstance(args[0], list) == False:
            self.func["expression"]["variable"].append(args[0])
        if isinstance(args[1], float) == False and isinstance(args[1], list) == False:
            self.func["expression"]["variable"].append(args[1])
        return [args[0], "+", args[1]]

    def mul(self, args):
        if isinstance(args[0], float) and isinstance(args[1], float):
            return args[0] * args[1]

        if isinstance(args[0], float) == False and isinstance(args[0], list) == False:
            self.func["expression"]["variable"].append(args[0])
        if isinstance(args[1], float) == False and isinstance(args[1], list) == False:
            self.func["expression"]["variable"].append(args[1])
        return [args[0], "*", args[1]]

    def sub(self, args):
        if isinstance(args[0], float) and isinstance(args[1], float):
            return args[0] - args[1]

        if isinstance(args[0], float) == False and isinstance(args[0], list) == False:
            self.func["expression"]["variable"].append(args[0])
        if isinstance(args[1], float) == False and isinstance(args[1], list) == False:
            self.func["expression"]["variable"].append(args[1])
        return [args[0], "-", args[1]]

    def neg(self, args):
        if isinstance(args[0], float):
            return -args[0]

        if isinstance(args[0], float) == False and isinstance(args[0], list) == False:
            self.func["expression"]["variable"].append(args[0])
        return [args[0], "-"]

    def var(self, args):
        if isinstance(args[0], float):
            return args[0]

        self.func["expression"]["variable"].append(args[0])
        return args


def calc_expression(variables: dict, operations: list):
    def substitute_var(elemento):
        if isinstance(elemento, list):
            return [substitute_var(e) for e in elemento]
        elif elemento in variables:
            return variables[elemento]
        else:
            return elemento

    operations_sostituite = substitute_var(operations)
    # print(operations_sostituite)

    def esegui_operations(operations):
        if isinstance(operations[0], list):
            risultato_1 = esegui_operations(operations[0])
        else:
            risultato_1 = operations[0]

        if len(operations) == 2:
            if operations[1] == "-":
                return -risultato_1

        if len(operations) == 3:
            if isinstance(operations[2], list):
                risultato_2 = esegui_operations(operations[2])
            else:
                risultato_2 = operations[2]

            if operations[1] == "*":
                return risultato_1 * risultato_2
            elif operations[1] == "+":
                return risultato_1 + risultato_2
            elif operations[1] == "/":
                return risultato_1 / risultato_2
            elif operations[1] == "-":
                return risultato_1 - risultato_2
            else:
                print(f"Invalid operation!! {operations[1]}")
        else:
            return risultato_1

    risultato_finale = esegui_operations(operations_sostituite)
    return risultato_finale


if __name__ == "__main__":
    # Example function definition and call
    function_definition = """ 
        function add(x, y, z) { return -x+2-y*z; }
        add(3, 4, 2);
    """

    # function_call = "add (3, 4);"

    # Create the Lark parser
    # parser = Lark(grammar, parser="lalr", start="start", transformer=FunctionTransformer())
    parser = Lark.open(
        "prova.lark",
        rel_to=__file__,
        start="start",
        parser="lalr",
        transformer=FunctionTransformer(),
    )
    # Parse the function definition and call
    parsed_function = parser.parse(function_definition)

    # function_call = parser.parse("add(3, 4, 2);")

    # parsed_call = parser.parse(function_call)

    # Print the parsed results
    print("\nFunction Definition:", parsed_function)
    # print("Function Call:", parsed_call)
