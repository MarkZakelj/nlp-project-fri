from doctest import OPTIONFLAGS_BY_NAME
from attr import attr
from nltk.stem import PorterStemmer, WordNetLemmatizer
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
import csv
import os

from sklearn.model_selection import PredefinedSplit
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

################### INITIALIZATION AND READING FILES ###################
# READ GT AND PRED FILES
root = os.getcwd()
experiment_paths = [os.path.join(root,"data/experiments",f) for f in  \
                    os.listdir(os.path.join(root,"data/experiments")) \
                    if os.path.isdir(os.path.join(root,"data/experiments",f))]
for experiment_path in experiment_paths:
    gt_path = os.path.join(experiment_path, "test.csv")
    model_dirs = [f for f in os.listdir(experiment_path) if os.path.isdir(f)]
    for model_dir in model_dirs:
        pred_path = os.path.join(experiment_path,model_dir,"annotation.csv")

model_dir = "bert-base-cased"
gt_path = os.path.join(experiment_paths[1], "test.csv")
pred_path = os.path.join(experiment_paths[1], model_dir,"annotation.csv")

# GT FILE
file = open(gt_path)
csvreader = csv.reader(file)
header = next(csvreader)
gt_data = []
for row in csvreader:
    gt_data.append(row)
file.close()

# PRED FILE
file = open(pred_path)
csvreader = csv.reader(file)
header = next(csvreader)
pred_data = []
for row in csvreader:
    pred_data.append(row)
file.close()

########################## PARAMETERS ##########################
SENT_COL = header.index('Sentence')
WORD_COL = header.index('Word')
TAG_COL = header.index('Tag')

second_tag = 'GENUS'

####################### DEFINED FUNCTIONS #######################
def get_tag_masks(data):
    mask_definiendum, mask_second_tag = [],[]
    for row in data:
        mask_definiendum.append(['DEFINIENDUM' in tag for tag in row][TAG_COL])
        mask_second_tag.append([second_tag in tag for tag in row][TAG_COL])
    return mask_definiendum, mask_second_tag

def get_stems_lemmas(data, mask):
    stem = ''
    lemma = ''
    stems = []
    lemmas = []
    for i, is_tag in enumerate(mask) :
        if is_tag:
            stem += ' ' + ps.stem(data[i][WORD_COL])
            lemma += ' ' + lemmatizer.lemmatize(data[i][WORD_COL])
            sent_nr = gt_data[i][SENT_COL] # serves as a node number
        elif stem != '':
            stems.append((sent_nr, stem.strip().lower()))
            lemmas.append((sent_nr, lemma.strip().lower()))
            stem = ''
            lemma = ''
    return stems, lemmas

def corrected_partial_overlap(gt, correct):
    corrected = [(i, gt_key) for (i, gt_key) in gt for (j, c_key) in correct if \
                    (i == j and c_key in gt_key)]
    return corrected

def generate_nodes_and_edges_list(definiendum_stems, definiendum_lemmas, \
                                  genus_stems, genus_lemmas):
    # GENERATE NODES AND EDGES
    edge_list = []
    nodes = {} # name of the node will be first lemma assigned to a stem
    for d_num, (i, d_stem) in enumerate(definiendum_stems):

        definiendum_lemmas[d_num]
        if d_stem not in nodes:
            nodes[d_stem] = {"name": definiendum_lemmas[d_num][1]}

        for g_num, g_stem in [(num, stem) for num, (k, stem) in enumerate(genus_stems) if i == k]:
            
            if g_stem not in nodes:
                nodes[g_stem] = {"name": genus_lemmas[g_num][1]}
            edge_list.append((d_stem,g_stem))
            
    return edge_list, nodes

def generate_graph_with_node_names(edge_list, nodes):
    G = nx.MultiDiGraph()
    G.add_nodes_from([(stem, dict_name) for stem, dict_name in nodes.items()])
    G.add_edges_from(edge_list)

    node_names = {}
    for stem, attr in G.nodes(data=True):
        node_names[stem] = attr["name"]
    
    return G, node_names

def plot_graph(G, G_correct, G_false, gt_node_names, correct_node_names, false_node_names):

    fig = plt.figure(figsize=(20, 20))
    layout = nx.spring_layout(G, seed=3113794657)
    layout_f = nx.spring_layout(G_false, seed=3113794657)
    #nx.draw_networkx_nodes(G, layout, node_color=colors)
    # nodes
    options_ = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.6}
    options = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.9}

    nx.draw_networkx_nodes(G, layout, nodelist=G.nodes, node_color="tab:gray", **options_) #gt noeds
    nx.draw_networkx_nodes(G_false, layout_f, nodelist=G_false.nodes, node_color="tab:red", **options_) #false nodes
    nx.draw_networkx_nodes(G, layout, nodelist=G_correct.nodes, node_color="tab:green", **options) #correct nodes
    nx.draw_networkx_edges(G, layout, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, layout, gt_node_names, font_size=16, font_color="gray")
    nx.draw_networkx_labels(G, layout, correct_node_names, font_size=16, font_color="green")
    nx.draw_networkx_labels(G_false, layout_f, false_node_names, font_size=10, font_color="red")
    # plt.tight_layout()
    # plt.axis("off")
    fig.savefig("graph.png")

if __name__ == '__main__':

    gt_mask_definiendum, gt_mask_genus = get_tag_masks(gt_data)
    pred_mask_definiendum, pred_mask_genus = get_tag_masks(pred_data)
    correct_mask_definiendum = [m1 and m2 for m1,m2 in zip(gt_mask_definiendum, pred_mask_definiendum)]
    false_mask_definiendum = [True if m1 is True and m2 is False else False for m1,m2 in zip(gt_mask_definiendum, pred_mask_definiendum)]
    correct_mask_genus = [m1 and m2 for m1,m2 in zip(gt_mask_genus, pred_mask_genus)]
    false_mask_genus = [True if m1 is True and m2 is False else False for m1,m2 in zip(gt_mask_genus, pred_mask_genus)]

    gt_definiendum_stems, gt_definiendum_lemmas = get_stems_lemmas(gt_data, gt_mask_definiendum)
    correct_definiendum_stems, correct_definiendum_lemmas = get_stems_lemmas(gt_data, correct_mask_definiendum)
    false_definiendum_stems, false_definiendum_lemmas = get_stems_lemmas(gt_data, false_mask_definiendum)

    gt_genus_stems, gt_genus_lemmas = get_stems_lemmas(gt_data, gt_mask_genus)
    correct_genus_stems, correct_genus_lemmas = get_stems_lemmas(gt_data, correct_mask_genus)
    false_genus_stems, false_genus_lemmas = get_stems_lemmas(gt_data, false_mask_genus)    

    # modify correct difiniendums and genus to match those of GT (if there is partial overlap)
    correct_definiendum_stems = corrected_partial_overlap(gt_definiendum_stems, correct_definiendum_stems)
    correct_definiendum_lemmas = corrected_partial_overlap(gt_definiendum_lemmas, correct_definiendum_lemmas)
    correct_genus_stems = corrected_partial_overlap(gt_genus_stems, correct_genus_stems)
    correct_genus_lemmas = corrected_partial_overlap(gt_genus_lemmas, correct_genus_lemmas)

    gt_edge_list, gt_nodes =  generate_nodes_and_edges_list(gt_definiendum_stems, gt_definiendum_lemmas, \
                                                        gt_genus_stems, gt_genus_lemmas)
    correct_edge_list, correct_nodes =  generate_nodes_and_edges_list(correct_definiendum_stems, correct_definiendum_lemmas, \
                                                        correct_genus_stems, correct_genus_lemmas)
    false_edge_list, false_nodes =  generate_nodes_and_edges_list(false_definiendum_stems, false_definiendum_lemmas, \
                                                        false_genus_stems, false_genus_lemmas)


    G_gt, gt_node_names = generate_graph_with_node_names(gt_edge_list, gt_nodes)
    G_correct, correct_node_names = generate_graph_with_node_names(correct_edge_list, correct_nodes)
    G_false, false_node_names = generate_graph_with_node_names(false_edge_list, false_nodes)

    # print(gt_node_names)
    # print(correct_node_names)

    plot_graph(G_gt, G_correct, G_false, gt_node_names, correct_node_names, false_node_names)


