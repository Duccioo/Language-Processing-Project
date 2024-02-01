# create  vitual environment: virtualenv -p python .venv .
# activate virtual environment: .\.venv\Scripts\activate 


from lark import Lark


from parser_functions import parser_from_file, parser_from_string
from visualizer_functions import create_pdf_and_svg
from Printer import Printer

PRINT_FIRST_PART_EXERCISE    = True
PRINT_SECOND_PART_EXERCISE   = True
CREATE_PDF_AND_SVG      = False 
GRAMMAR_FROM_FILE       = False 
INPUT_STRING            = 'x=1;y=2;switch(x){case 0:z=10;break;default:z=1;break;}'
GRAMMAR_FILE            = 'grammar.lark'
INLINE_GRAMMAR          =  r'''
                                ?operations:(assignments)+(switch_statement)
                                switch_statement: ((switch_declaration) (execution_block))*
                                switch_declaration:  (condition)
                                condition:  CONDITION_KEYWORD "(" VARIABLES ")" 
                                        |   CONDITION_KEYWORD "(" NUMBER ")"
                                        |   CONDITION_KEYWORD "()"
                                execution_block: "{" (case_block)* (default_block) "}"
                                case_block: "case" NUMBER ":" assignment_result break_switch 
                                default_block: "default:" assignment_result break_switch
                                break_switch: "break;"
                                assignments: VARIABLES "=" NUMBER ";" | VARIABLES ";"
                                assignment_result: RESULT_VARIABLE "=" NUMBER ";"
                                CONDITION_KEYWORD : "switch" | "if" | "else"
                                VARIABLES: "x"|"y"|"X"|"Y" 
                                RESULT_VARIABLE: "z"|"Z"
                                %import common.ESCAPED_STRING
                                %import common.NUMBER
                                %import common.WS_INLINE
                                %ignore WS_INLINE 
                            '''


if __name__ == "__main__":
    ########### PRIMA PARTE DELL'ESERCIZIO ############
    if PRINT_FIRST_PART_EXERCISE:

        # Versione con Grammatica su stringa inline in python
        parser = parser_from_file(GRAMMAR_FILE) if GRAMMAR_FROM_FILE else parser_from_string(INLINE_GRAMMAR)
        parse_tree = parser.parse(INPUT_STRING)
        # stampa il parse tree
        print(parse_tree.pretty())
        # crea i file pdf e svg del parse tree
        if CREATE_PDF_AND_SVG:create_pdf_and_svg(parse_tree, INPUT_STRING, "pdf", "svg") 


    ########### SECONDA PARTE DELL'ESERCIZIO ############
    if PRINT_SECOND_PART_EXERCISE:
       
        transformer = Printer()
        parser = parser_from_file(GRAMMAR_FILE, transformer = transformer) \
            if GRAMMAR_FROM_FILE \
            else parser_from_string(INLINE_GRAMMAR, transformer = transformer)
       
        result = parser.parse(INPUT_STRING)
        print(f'Result of the parser:\t{result}')
        