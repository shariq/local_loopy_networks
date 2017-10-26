# the goal of all this complicated stuff is to make sampling fast and easy

# when sampling, we pick a type first then ask for something of that type

# using .ne extension to get syntax highlighting of nearly CFG files; but we don't care at all about parsing, we only care about string generation

### CONSTANTS ###

# evaluated first and does some replacements

CONSTDEF EDGE_MEMORY_SIZE -> 1
CONSTDEF FILTER_COUNT -> 5


### TYPES ###

# `' | '.join('_edge_memory_{}'.format(i) for i in range(EDGE_MEMORY_SIZE))` is run as python code and evaluated

# anything in {} is a python macro which is pure; takes in a string and returns a string

# hierarchial types are syntax sugar; they actually are used to generate a bunch of other rules which use primitive types

# $ types are used at runtime for searching

# all types must start with _

TYPEDEF _type -> _float | _vector

TYPEDEF _vector ->
    _signal | _error |
    `'|'.join('_edge_memory_{}'.format(i) for i in range(EDGE_MEMORY_SIZE))` |
    `'|'.join('_filter_{}'.format(i) for i in range(FILTER_COUNT))`


### RULES ###

# _$0 means type of 0th index unbound variable in expression

# _$0,1 means type of 0th index unbound variable and 1st index unbound variable have to be the same, and that's the type of this expression

# _$$0 means type of 0th index at compile time in expression; before generating anything

_type start -> float | vector

_float float_base -> "0.0" | "-1.0" | "1.0" | "-2.0" | "2.0" | "0.1"
_float float_expression -> multiply | add
_float float -> float_base | float_expression

_vector vector_base ->
    _signal "signal" |
    _error "error" |
    _$$0 edge_memory |
    _$$0 filter

_type edge_memory -> `'|'.join('_edge_memory_{i} "edge_memory_{i}"'.format(i=i) for i in range(EDGE_MEMORY_SIZE))`

_vector vector_expression -> multiply | add
_vector vector -> multiply | add

_type multiply -> _float "multiply(" float ", " float ")" | _$0 "multiply(" vector ", " float ")" | _$0$1 "multiply(" vector ", " vector ")"

_type add -> _float "add(" float ", " float ")" | _$0 "add(" vector ", " float ")" | _$0,1 "add(" vector ", " vector ")"

_float sum_vector -> "sum(" vector ")"
