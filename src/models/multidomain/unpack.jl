export @unpack_params

"""
    @unpack_params a,b,c,... = m(p)

Extracts the parameters `a,b,c,...` from the model `m` evaluated at the domain `p`.
This is a macro that essentially lowers to 
```julia
a = getparam(m, :a, p)
b = getparam(m, :b, p)
...
```
For any model that may depend on a `DomainParams` type this macro should be used to 
extract the parameters. 

!!! warn
    This feature is experimental and is not considered part of the public stable API.

"""
macro unpack_params(args)
    args.head != :(=) &&
        throw(ArgumentError("Expression needs to be of the form a, b, = c(p)"))
    items, suitcase = args.args
    items = isa(items, Symbol) ? [items] : items.args
    hasproperty(suitcase, :head) ||
        throw(ArgumentError("RHS of expression must be of form m(p)"))
    suitcase.head != :call && throw(ArgumentError("RHS of expression must be of form m(p)"))
    m, p = suitcase.args[1], suitcase.args[2]
    paraminstance = gensym()
    kp = [
        :($key = getparam($paraminstance, Val{$(Expr(:quote, key))}(), $p))
            for key in items
    ]
    kpblock = Expr(:block, kp...)
    expr = quote
        local $paraminstance = $m
        $kpblock
        $paraminstance
    end
    return esc(expr)
end
