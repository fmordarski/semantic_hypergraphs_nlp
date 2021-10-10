using  SimpleHypergraphs
import LightGraphs


function prepare_input(doc)
    data = Matrix{Union{Missing, String}}(undef, 0, 7)
    for i in 1:length(doc)
        token = doc[i]
        if i != 1
            previous = doc[i - 1].pos_
        else
            previous = "None"
        end
        row = [token.pos_, token.tag_, token.dep_, token.head.pos_,
               token.head.tag_, token.head.dep_, previous]
        data = vcat(data, reshape(row, (1, 7)))
    end
    return data
end

function alpha(doc, model)
    input = prepare_input(doc)
    return predict(model, input)
end

pattern_1 = ("M", "x")
pattern_2 = ("B", "C", "C", "+")
pattern_3 = ("T", ["C", "R"])
pattern_4 = ("P", ["C", "R", "S"], "+")
pattern_5 = ("J", "x")
patterns = [pattern_1, pattern_2, pattern_3, pattern_4, pattern_5]

function find_pattern(pattern, seq)
    if pattern[1] in seq
        if (pattern[1] == "M") & (length(seq) == 2)
            if seq[1] != "M"
                return([seq[2], seq[1]], seq[1], [2, 1])
            else
                return(seq, seq[2], [1, 2])
            end
        elseif (pattern[1] == "B") & (length(seq) > 2)
            rest = filter(e->e≠"B", seq)
            if ((length(rest) + 1) == length(seq)) & (all(y->y == "C", rest))
                order = findall(isequal("B"), seq)
                return(append!(["B"], rest), "C", append!(order, [i for i in 1:length(seq) if i != order[1]]))
            end
        elseif (pattern[1] == "T") & (length(seq) == 2)
            rest = filter(e->e≠"T", seq)
            if (length(rest) == 1) & (rest[1] in ["C", "R"])
                order = findall(isequal("T"), seq)
                return(append!(["T"], rest), "S", append!(order, [i for i in 1:length(seq) if i != order[1]]))
            end
        elseif (pattern[1] == "P")
            rest = filter(e->e≠"P", seq)
            if (all(y->y in ["C", "S", "R"], rest)) & ((length(rest) + 1) == length(seq))
                order = findall(isequal("P"), seq)
                return(append!(["P"], rest), "R", append!(order, [i for i in 1:length(seq) if i != order[1]]))
            end
        elseif (pattern[1] == "J")
            rest = filter(e->e≠"J", seq)
            if (length(rest) == 2)
                order = [findfirst(isequal("J"), seq)]
                return(append!(["J"], rest), rest[1], append!(order, [i for i in 1:length(seq) if i != order[1]]))
            end
        end
    end
end

function function_h(hg, doc)
    tokens = [token for token in doc]
    texts = [token.text for token in tokens]
    score = 0
    global ones_hg = [[i for i in 1:nhv(hg) if hg[i, e] == 1] for e in 1:nhe(hg)]
    connected_vertices = [i for i in 1:nhv(hg) if 
                         length([e for e in 1:nhe(hg) if hg[i, e] == 1]) > 2]
    if length(connected_vertices) < 1
        score = 1
        for v in 1:length(ones_hg)
            tokens_new = tokens[ones_hg[v]]
            depths = [length([ancestor.text for ancestor in token.ancestors]) 
                                               for token in tokens_new]
            min_depths = findall(isequal(minimum(depths)), depths)
            min_depth_token = tokens_new[findfirst(isequal(minimum(depths)), depths)]
            rest = [token for token in tokens_new if token != min_depth_token]
            if length(min_depths) > 1
                return 0
            end
            childs = []
            for token in tokens_new
                children_token = [child for child in token.children]
                append!(childs, children_token)
            end
            if any(y->y ∉ childs, rest)
                return 0
            end
        end
        return score
    end
    for e in 1:length(ones_hg)
        check = []
        for rest_e in 1:length(ones_hg)
            if rest_e == e
                break
            end
            head_indexes = [findfirst(isequal(token.head.text), texts) for token in tokens[ones_hg[rest_e]]]
            child_indexes = [findfirst(isequal(child), tokens) 
                            for token in tokens[ones_hg[rest_e]] for child in token.children]
            if any(y->y in ones_hg[e], vcat(head_indexes, child_indexes))
                append!([true], check)
            else
                append!([false], check)
            end
        end
        if all(check)
            score += 1
        end
    end
    return score
end     

function find_original(hg, indexes)
    if all(y->y == nothing, hg)
        return indexes
    end
    mapping = Dict()
    i = 1
    connected_vertices = []
    for v in 1:nhv(hg)
        if v in connected_vertices
            mapping[i] = (minimum(connected_vertices), maximum(connected_vertices))
            continue
        end
        if all(y->y == nothing, hg[v, :])
            mapping[i] = (v, v)
        else
            edge = maximum([e for e in 1:length(hg[v, :]) if hg[v, e] == 1])
            connected_vertices = [v]
            for vertice in (v+1):length(hg[:, edge])
                if hg[vertice, edge] == nothing
                    break
                end
                append!(connected_vertices, vertice)
            end
            mapping[i] = (minimum(connected_vertices), maximum(connected_vertices))
        end
        i += 1
    end
    first = mapping[indexes[1]][1]
    last = mapping[indexes[2]][2]
    return (first, last)
end

function get_copy_hg(hg)
    return Hypergraph(copy(hg))
end

function get_depth(hg, depths, indexes)
    valid_indexes = []
    for i in indexes[1]:indexes[2]
        if all(y->y == nothing, hg[i, :])
            append!(valid_indexes, i)
        end
    end
    if length(valid_indexes) < 1
        return(mean(depths[indexes[1]:indexes[2]]))
    else
        return mean(depths[valid_indexes])
    end
end
        

function parsing(hg, parsed_text, order)
    output = Dict()
    ones_hg = [[i for i in 1:nhv(hg) if hg[i, e] == 1] for e in 1:nhe(hg)]
    for e in 1:nhe(hg)
        output[e] = Any[]
        if e != 1
            connected_edges = [edge for edge in 1:e-1 
                               if all(y->y in ones_hg[e], ones_hg[edge])]
            if length(connected_edges) == 0
                indexes = [i + ones_hg[e][1] - 1 for i in order[e]]
                output[e] = parsed_text[indexes]
            elseif length(connected_edges) == 1
                closest_edge = output[connected_edges[1]]
                rest = [index for index in ones_hg[e] 
                        if index ∉ ones_hg[connected_edges[1]]]
                for o in order[e][1:length(rest)]
                    push!(output[e], parsed_text[rest][o])
                end
                push!(output[e], closest_edge)
            else
                vertices_edges = []
                for edge in connected_edges
                    append!(vertices_edges, ones_hg[edge])
                end
                edges_check = [edge for edge in 1:e-1 
                               if all(y->y in ones_hg[edge], vertices_edges)]
                if length(edges_check) > 0
                    closest_edge = output[maximum(edges_check)]
                    rest = [index for index in ones_hg[e] 
                            if index ∉ ones_hg[maximum(edges_check)]]
                    for o in order[e][1:length(rest)]
                        push!(output[e], parsed_text[rest][o])
                    end
                    push!(output[e], closest_edge)
                else
                    rest = [index for index in ones_hg[e]
                            if index ∉ vertices_edges]
                    for atom in rest
                        push!(output[e], parsed_text[atom])
                    end
                    for edge in connected_edges
                        push!(output[e], output[edge])
                    end
                end
            end
        else
            indexes = [(i + ones_hg[e][1] - 1) for i in order[e]]
            output[e] = parsed_text[indexes]
        end
    end
    return output
end 

function beta(patterns, doc, atoms)
    global tokens = [token for token in doc]
    global depths = [length([ancestor.text for ancestor in token.ancestors]) 
                                           for token in tokens]
    global atoms_depths = [[atom for atom in atoms], [depth for depth in depths]]
    global parsed_text = [(atom, token.text) for (token, atom) in zip(doc, atoms)]
    global hypergraph = Hypergraph{Float64}(length([token for token in doc]), 1)
    global order = []
    while any(y->y != 1, hypergraph[:, nhe(hypergraph)])
        global depth_best = 0
        global h_best = 0
        global count_best = 0
        found_patterns = []
        for pattern in patterns
            for i in 1:length(atoms_depths[1])
                start = i+1
                for j in start:length(atoms_depths[1])
                    output = find_pattern(pattern, atoms_depths[1][i:j])
                    if output != nothing
                        count = length(output[1])
                        append!(found_patterns, output)
                        orig_indexes = find_original(hypergraph, (i, j))
                        depth = get_depth(hypergraph, depths, orig_indexes)
                        copy_hg = get_copy_hg(hypergraph)
                        copy_hg[orig_indexes[1]:orig_indexes[2], nhe(copy_hg)] .= 1
                        h = function_h(copy_hg, doc)
#                         if (h > h_best) | ((h == h_best) & ((depth > depth_best))) |
#                            ((h == h_best) & (depth == depth_best) & (count > count_best))
                            
                        if (h > h_best) | ((h == h_best) & ((depth > depth_best)))
                            global best_output = output
                            global depth_best = depth
                            global best_pattern = output
                            global indexes = (i, j)
                            global orig_best_index = orig_indexes
                            global h_best = h
                            global count_best = count
                        end
                    end
                end
            end
        end
        if length(found_patterns) == 0
            first_index = floor(length(atoms_depths[1])/2)
            global output = (append!(["J"], atoms_depths[1]), "J", [i for i in 1:length(atoms_depths[1])])
            global orig_best_index = (1, nhv(hypergraph))
            global indexes = (1, length(atoms_depths[1]))
        end
        hypergraph[orig_best_index[1]:orig_best_index[2], nhe(hypergraph)] .= 1
        atoms_depths[1][indexes[1]:indexes[2]] .= best_pattern[2]
        atoms_depths[2][indexes[1]:indexes[2]] .= maximum(atoms_depths[2][indexes[1]:indexes[2]])
        deleteat!(atoms_depths[1], indexes[1]+1:indexes[2])
        deleteat!(atoms_depths[2], indexes[1]+1:indexes[2])
        append!(order, [best_output[3]])
        if any(y->y != 1, hypergraph[:, nhe(hypergraph)])
            add_hyperedge!(hypergraph)
        end
    end
    return (hypergraph, parsing(hypergraph, parsed_text, order))
end
