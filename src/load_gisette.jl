####gisette
#https://archive.ics.uci.edu/dataset/170/gisette

using SparseArrays
import Random 

function load_gisette(;filename ="gisette", sparse_data = true, do_normalize = false)

    f = open("socgs_data/"*filename*"_train.data")
    Z_data = []


    while ! eof(f)
    line = readline(f)
    spl_line = split(line)
    push!(Z_data,parse.(Float64,spl_line))
    end
    m = length(Z_data)
    n = length(Z_data[1])
    Z = Matrix{Float64}(undef,n,m)
    for j in 1:m
        for i in 1:n        
            Z[i,j] = Z_data[j][i]
        end
    end
    if sparse_data
        Z = sparse(Z)
    end
    close(f)

    if do_normalize
        means = sum(Z,dims=2)
        means ./=m
        Z .-= means
        vars = sum(abs2,Z,dims=2)
        vars ./= (m-1)
        vars[findall(==(zero(eltype(vars))),vars)] .= one(eltype(vars))
        Z./=sqrt.(vars)
    end

    f = open("socgs_data/"*filename*"_train.labels")
    y = Vector{Float64}(undef,m)
    i = 0
    while ! eof(f)
        i +=1
        line = readline(f)
        y[i] = parse(Float64,line)
    end
    close(f)

    return y,Z
end

function sample_gisette(;seed = 222,feature_sample_size = 100, row_sample_size = 100)
    y,Z = load_gisette()
    n,m = size(Z)
    @assert 1<= feature_sample_size <= n
    @assert 1<= row_sample_size <= m
    Random.seed!(seed)
    if feature_sample_size ==n 
        choice_features = 1:n
    else
        choice_features = rand(1:n,feature_sample_size)
    end
    if row_sample_size == m
        choice_features = 1:m
    else
        choice_rows = rand(1:m,row_sample_size)
    end
    sampled_Z = Z[choice_features,choice_rows]
    sampled_y = y[choice_rows]
    return sampled_y,sampled_Z
end
