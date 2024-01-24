from lark import Lark
import os
from Printer import Printer

########### FUNZIONI PER PRIMA PARTE DELL'ESERCIZIO ############
def parser_from_string(grammar_string : str, start_block : str = "operations", transformer : Printer = None):
    assert len(grammar_string) > 0, "Grammar string is empty"
    config = {
        'grammar' : grammar_string, 
        'start' : start_block,
    }
    if transformer is None:
        config = {
            'ambiguity' : "explicit", 
            'parser' : 'earley',
            **config}
    else:
        config = {
            'transformer' : transformer, 
            'parser' : 'lalr',
            **config}
    parser = Lark(**config)
    return parser

def parser_from_file(grammar_file : str, start_block : str = "operations", transformer : Printer = None):
    assert os.path.exists(grammar_file) and os.path.isfile(grammar_file), f"File {grammar_file} does not exist"
    config = {
        'grammar_filename' : grammar_file, 
        'start' : start_block,
    }
    if transformer is None:
        config = {
            'ambiguity' : "explicit", 
            'parser' : 'earley',
            **config}
    else:
        config = {   
            'transformer' : transformer, 
            'parser' : 'lalr',
            **config}
    parser = Lark.open(**config)
    return parser





#Se non passi un transformer, la funzione utilizza il parser Earley e imposta il livello di ambiguità su "explicit" per gestire eventuali ambiguità nella grammatica. Questa scelta potrebbe essere basata sulla considerazione che il parser Earley è in grado di gestire grammatiche più generiche e ambigue rispetto al parser LALR, ma richiede più risorse computazionali.
#Se passi un transformer, la funzione utilizza il parser LALR. Questa scelta potrebbe essere basata sulla considerazione che il parser LALR è spesso più veloce nell'analisi sintattica rispetto al parser Earley e potrebbe essere sufficiente per grammatiche meno ambigue. 