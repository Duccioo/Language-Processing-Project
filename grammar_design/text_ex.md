## COMMAND:

Using lark implement a parser for the definition of functions, with the following
rules

- the functions are defined as

  function name(par1,par2,…) {
  return par1 op par2 op par3…;
  }

where name is the function name with the usual restrictions (an alphanumeric string
beginning with a letter), par1.. are the function parameters whose names follow the
same rules as variables names, op is + or \* (sum or product). The function body contains
only the return instruction that involves the parameters.

- assume that only one function can be defined
- after the function definition, there are the calls whose syntax is

  name(cost1,cost2,…);

  where name is the name of a defined function, cost1,… are numeric constants in the same
  number as the function arguments.

- print the result of each function call

## PLUS

Puoi eventualmente estendere al caso in cui puoi definire più di una funzione.
Si complica solo leggermente il codice da eseguire per la memorizzazione
delle funzioni definite.
