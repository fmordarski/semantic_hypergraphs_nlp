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

doc = nlp("Berlin is the capital of Germany")

hg, _ = beta(patterns, doc, alpha(doc, rf))

@testset "Hypergraphs tests" begin
    doc = nlp("Berlin is the capital of Germany")
    hg, _ = beta(patterns, doc, alpha(doc, rf))


    doc = nlp("Berlin is the capital of Germany")
    hg, _ = beta(patterns, doc, alpha(doc, rf))


    doc = nlp("Berlin is the capital of Germany")
    hg, _ = beta(patterns, doc, alpha(doc, rf))


    doc = nlp("Berlin is the capital of Germany")
    hg, _ = beta(patterns, doc, alpha(doc, rf))


    doc = nlp("Berlin is the capital of Germany")
    hg, _ = beta(patterns, doc, alpha(doc, rf))


    doc = nlp("Berlin is the capital of Germany")
    hg, _ = beta(patterns, doc, alpha(doc, rf))


    doc = nlp("Berlin is the capital of Germany")
    hg, _ = beta(patterns, doc, alpha(doc, rf))


    doc = nlp("Berlin is the capital of Germany")
    hg, _ = beta(patterns, doc, alpha(doc, rf))


