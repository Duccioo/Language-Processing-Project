from lark import Transformer

########### CLASSE PER SECONDA PARTE DELL'ESERCIZIO ############
class Printer(Transformer):
    NUMBER = int
    DEFAULT_ASSIGNMENT_VALUE : int = 0
    ERROR_MESSAGE : str = 'One error occurred during the execution of the switch statement'
    
    def __init__(self, debug = False):
        self.vars = {}
        self.case_assignments = {}
        self.result = {}
        self.DEBUG = debug
    def operations(self,args):
        '''Ritorniamo il risultato dello switch_statement'''
        return args[-1]
    
    def assignments(self, args):
        if len(args) == 1:
            self.vars[args[0]] = Printer.DEFAULT_ASSIGNMENT_VALUE
        else:
            self.vars[args[0]] = args[1]
    
    def switch_statement(self, args):
        # ritorna il valore della condizione in maniera ricorsiva
        variableLabel = args[0]
        casesJson = args[1]
        valueOfVariable = self.vars[variableLabel]
        for case in casesJson:
            if case['case-number'] == 'default':
                self.result = case['assignment']
                case['break'] = True
                break
            if case['case-number'] == valueOfVariable:
                self.result = case['assignment']
                case['break'] = True
                break
            else:
                self.result = Printer.ERROR_MESSAGE
        
        return self.result if not self.DEBUG else self.result, casesJson

    def switch_declaration(self, args):
        # ritorna il valore della variabile usata nello switch
        return args[0]
    
    def condition(self, args):
        # Per come è definita la grammatica ritorna il nome della variabile che ispezioniamo
        return args[-1]
    
    def execution_block(self, args):
        '''
            Args dovrebbe essere una lista di json del tipo
            [
                {
                    'case-number' : <numero>, 
                    'assignment' : <numero>, 
                    'break' : False
                },
                {
                    'case-number' : <numero>, 
                    'assignment' : <numero>, 
                    'break' : False
                },
                {
                    'case-number' : 'default', 
                    'assignment' : <numero>, 
                    'break' : False
                },
            ]
        '''
        return args
        
    def case_block(self, args):
        '''
        riceverà 
            -   il numero legato al caso dal TOKEN_NUMBER
            -   il numero dato in assegnazione a Z da assignment_result
            -   False dal break_switch
        '''
        return {'case-number' : args[0], 'assignment' : args[1], 'break' : args[2]}
    
    def default_block(self, args):
        return {'case-number' : 'default', 'assignment' : args[0], 'break' : args[1]}
    
    def break_switch(self,args):return False
    
    def assignment_result(self, args):
        return args[1] if len(args) == 2 else Printer.DEFAULT_ASSIGNMENT_VALUE 
