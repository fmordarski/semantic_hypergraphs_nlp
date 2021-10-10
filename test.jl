using PyCall, BSON, DecisionTree, Random, Statistics, ArgParse, SimpleHypergraphs, Test

import LightGraphs

include("./main_parser.jl")

s = ArgParseSettings()
@add_arg_table s begin
    "--model"
        help = "Path to first stage model"
        arg_type = String
        default = "models/model_rf.bson"
end

parsed_args = parse_args(ARGS, s)

BSON.@load parsed_args["model"] rf

spacy = pyimport("spacy")

nlp = spacy.load("en_core_web_lg")

@testset "Hypergraphs tests" begin
    doc = nlp("Berlin is the capital of Germany")
    hg, _ = beta(patterns, doc, alpha(doc, rf))
    expected = Hypergraph{Float64}(6, 3); expected[4:6, 1] .= 1
    expected[3:6, 2] .= 1; expected[1:6, 3] .=1
    @test hg == expected

    doc = nlp("Mary likes astronomy and plays football")
    hg, _ = beta(patterns, doc, alpha(doc, rf))
    expected = Hypergraph{Float64}(6, 3); expected[5:6, 1] .= 1
    expected[1:3, 2] .= 1; expected[1:6, 3] .=1
    @test hg == expected

    doc = nlp("Alice says dogs are nice")
    hg, _ = beta(patterns, doc, alpha(doc, rf))
    expected = Hypergraph{Float64}(5, 3); expected[3:5, 1] .= 1
    expected[1:2, 2] .= 1; expected[1:5, 3] .=1
    @test hg == expected

    doc = nlp("Bob wants to play chess")
    hg, _ = beta(patterns, doc, alpha(doc, rf))
    expected = Hypergraph{Float64}(5, 3); expected[3:4, 1] .= 1
    expected[3:5, 2] .= 1; expected[1:5, 3] .=1
    @test hg == expected

    doc = nlp("Alice says dogs are nice")
    hg, _ = beta(patterns, doc, alpha(doc, rf))
    expected = Hypergraph{Float64}(5, 3); expected[3:5, 1] .= 1
    expected[1:2, 2] .= 1; expected[1:5, 3] .=1
    @test hg == expected
end
