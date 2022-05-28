from doctest import OPTIONFLAGS_BY_NAME
from pkgutil import extend_path
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
import warnings

########################## PARAMETERS ##########################
SECOND_TAG = 'GENUS'
MAX_WCC_TO_DISPLAY = 3
################################################################


def get_path(experiment_path, prev_mo):
    pred_path = None 
    mo = None
    gt_path = os.path.join(experiment_path, "test.csv")
    model_dirs = [f for f in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path,f))]
    global ex_path
    ex_path = experiment_path

    for mo, model in enumerate(model_dirs):
        if prev_mo >= mo:
            continue
        elif prev_mo == len(model_dirs)-1:
            mo = -1
        else:
            print(experiment_path.split("/")[-1],"/",model)
            pred_path = os.path.join(experiment_path,model,"annotation.csv")
            global model_dir
            model_dir = model
            break

    return gt_path, pred_path, mo

def get_data(gt_path, pred_path):
    # GT FILE
    file1 = open(gt_path)
    csvreader1 = csv.reader(file1)
    header1 = next(csvreader1)
    gt_data = []

    # PRED FILE
    file2 = open(pred_path)
    csvreader2 = csv.reader(file2)
    header2 = next(csvreader2)

    if header1 == header2:
        header = header1
    else:
        warnings.warn("Warning: different headers in .csv files!")

    pred_data = []
    for row1,row2 in zip(csvreader1,csvreader2):
        if row1[header.index('Word')] != row2[header.index('Word')]:
            print(row1, row2)
            print("Error: words in rows are not the same!")
            return None, None
        else:
            gt_data.append(row1)
            pred_data.append(row2)
            
    file1.close()
    file2.close()

    global SENT_COL
    global WORD_COL
    global TAG_COL
    SENT_COL = header.index('Sentence')
    WORD_COL = header.index('Word')
    TAG_COL = header.index('Tag')

    return gt_data, pred_data

def get_tag_masks(data):
    mask_definiendum, mask_second_tag = [],[]
    for row in data:
        mask_definiendum.append(['DEFINIENDUM' in tag for tag in row][TAG_COL])
        mask_second_tag.append([SECOND_TAG in tag for tag in row][TAG_COL])
    return mask_definiendum, mask_second_tag

def get_gt_stems_lemmas(data, mask):
    stem = ''
    lemma = ''
    stems = []
    lemmas = []
    for i, is_tag in enumerate(mask):
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

def get_stems_lemmas(gt_data, masks, gt_stems, gt_lemmas):
    stems = []
    lemmas = []
    for i, is_tag in enumerate(masks):
        if is_tag:
            sent_nr = gt_data[i][SENT_COL] # serves as a node number
            word = lemmatizer.lemmatize(gt_data[i][WORD_COL]).strip().lower()
            if len(word) > 1:
                for j, (s, l) in enumerate(gt_lemmas):
                    # correct predictions
                    if word in l and sent_nr == s and \
                    gt_lemmas[j] not in lemmas and gt_stems[j] not in stems:
                        lemmas.append(gt_lemmas[j])
                        stems.append(gt_stems[j])               

    return stems, lemmas

def get_correct_and_false_data(gt,pred):
    correct = [(j, p_key) for (i, gt_key) in gt for (j, p_key) in pred if (i == j and gt_key == p_key)]
    false = [(j, p_key) for (i, gt_key) in gt for (j, p_key) in pred if (i == j and gt_key != p_key)]
    return correct, false

def corrected_partial_overlap(gt, correct):
    corrected = [(i, gt_key) if (i == j and c_key in gt_key) else (j, c_key) \
                 for (i, gt_key) in gt for (j, c_key) in correct]
    return corrected

def generate_nodes_and_edges_list(definiendum_stems, definiendum_lemmas, \
                                  genus_stems, genus_lemmas):
    # GENERATE NODES AND EDGES
    edge_list = []
    node_names = {} # name of the node will be first lemma assigned to a stem
    for d_num, (i, d_stem) in enumerate(definiendum_stems):

        if d_stem not in node_names:
            node_names[d_stem] = {"name": definiendum_lemmas[d_num][1]}

        # genuses that connect to the definienudm
        for g_num, g_stem in [(num, stem) for num, (k, stem) in enumerate(genus_stems) if i == k]:
            
            if g_stem not in node_names:
                node_names[g_stem] = {"name": genus_lemmas[g_num][1]}
            edge_list.append((d_stem,g_stem))

    # add all other genuses that are not connected to anything
    for g_num, (i, g_stem) in enumerate(genus_stems):
        if g_stem not in node_names:
            node_names[g_stem] = {"name": genus_lemmas[g_num][1]}

    return edge_list, node_names

def generate_graph_with_node_names(edge_list, nodes):
    G = nx.MultiDiGraph()
    G.add_nodes_from([(stem, dict_name) for stem, dict_name in nodes.items()])
    G.add_edges_from(edge_list)

    node_names = {}
    for stem, attr in G.nodes(data=True):
        node_names[stem] = attr["name"]
    
    return G, node_names

def plot_graph(G, G_correct, G_false, G_u, \
               correct_node_names, false_node_names, u_node_names):

    G.add_nodes_from(G_u.nodes(data=True))
    G.add_edges_from([e for e in G_u.edges])

    fig = plt.figure(figsize=(20, 20))
    layout = nx.spring_layout(G, seed=3113794657)
    #layout_f = nx.spring_layout(G_false, seed=3113794657)

    # options
    options_ = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.6}
    options = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.9}

    # nodes
    nx.draw_networkx_nodes(G, layout, nodelist=G_u.nodes, node_color="tab:gray", **options_) #gt nodes
    nx.draw_networkx_nodes(G, layout, nodelist=G_false.nodes, node_color="tab:red", **options_) #false nodes
    nx.draw_networkx_nodes(G, layout, nodelist=G_correct.nodes, node_color="tab:green", **options) #correct nodes
    # edges
    nx.draw_networkx_edges(G, layout, width=1, alpha=0.5)
    # nx.draw_networkx_edges(G, layout, edgelist=[e for e in G_pred.edges if e not in G_correct.edges], width=1, alpha=0.5, edge_color="tab:red")
    # nx.draw_networkx_edges(G, layout, edgelist=[e for e in G_correct.edges], width=1, alpha=0.5, edge_color="tab:green")
    # labels
    nx.draw_networkx_labels(G, layout, u_node_names, font_size=16, font_color="gray")
    nx.draw_networkx_labels(G, layout, correct_node_names, font_size=16, font_color="green")
    nx.draw_networkx_labels(G, layout, false_node_names, font_size=16, font_color="red")
    # plt.tight_layout()
    # plt.axis("off")
    fig.savefig(os.path.join(ex_path,model_dir,"graph.png"))

    # max wcc
    f = plt.figure(figsize=(20, 20))
    G_ = nx.MultiDiGraph()
    for i, max_wcc in enumerate(sorted(nx.weakly_connected_components(G), key=len, reverse=True)):
        if i in range(MAX_WCC_TO_DISPLAY):
            G_.add_nodes_from(nx.subgraph(G, max_wcc).nodes(data=True))
            G_.add_edges_from([e for e in nx.subgraph(G, max_wcc).edges])
        else:
            break
    
    G_mwcc = nx.subgraph(G, G_)
    pos = nx.spring_layout(G_mwcc, seed=311379)
    correct_wcc_names = {}
    false_wcc_names = {}
    for n, name in G_mwcc.nodes(data=True):
        if n in correct_node_names:
            correct_wcc_names[n] = name["name"]
        elif n in false_node_names:
            false_wcc_names[n] = name["name"]

    #print(correct_node_names)
    # options
    options_ = {"edgecolors": "tab:gray", "node_size": 1000, "alpha": 0.6}
    options = {"edgecolors": "tab:gray", "node_size": 1000, "alpha": 0.9}

    # nodes
    nx.draw_networkx_nodes(G_mwcc, pos, nodelist=[n for n in G_mwcc.nodes if n in G_false.nodes], node_color="tab:red", **options_) #false nodes
    nx.draw_networkx_nodes(G_mwcc, pos, nodelist=[n for n in G_mwcc.nodes if n in G_correct.nodes], node_color="tab:green", **options) #correct nodes
    # edges
    nx.draw_networkx_edges(G_mwcc, pos, width=7, alpha=0.5)
    # labels
    nx.draw_networkx_labels(G_mwcc, pos, correct_wcc_names, font_size=23, font_color="green")
    nx.draw_networkx_labels(G_mwcc, pos, false_wcc_names, font_size=23, font_color="red")

    f.savefig(os.path.join(ex_path,model_dir,"max_wcc_graph.png"))


def main(gt_data, pred_data):

    #for edges
    gt_mask_definiendum, gt_mask_genus = get_tag_masks(gt_data)
    pred_mask_definiendum, pred_mask_genus = get_tag_masks(pred_data)

    #for nodes
    correct_mask_definiendum = [True if (m1 == m2 and m1 == True) else False for m1,m2 in zip(gt_mask_definiendum, pred_mask_definiendum)]
    false_mask_definiendum = [True if (m1 != m2 and m2 == True) else False for m1,m2 in zip(gt_mask_definiendum, pred_mask_definiendum)]
    undetected_mask_definiendum = [True if (m1 != m2 and m2 == False) else False for m1,m2 in zip(gt_mask_definiendum, pred_mask_definiendum)]
    correct_mask_genus = [True if (m1 == m2 and m1 == True) else False for m1,m2 in zip(gt_mask_genus, pred_mask_genus)]
    false_mask_genus = [True if (m1 != m2 and m2 == True) else False for m1,m2 in zip(gt_mask_genus, pred_mask_genus)]
    undetected_mask_genus = [True if (m1 != m2 and m2 == False) else False for m1,m2 in zip(gt_mask_genus, pred_mask_genus)]

    #gt
    gt_definiendum_stems, gt_definiendum_lemmas = get_gt_stems_lemmas(gt_data, gt_mask_definiendum)
    gt_genus_stems, gt_genus_lemmas = get_gt_stems_lemmas(gt_data, gt_mask_genus)
    #pred
    pred_definiendum_stems, pred_definiendum_lemmas = get_gt_stems_lemmas(gt_data, pred_mask_definiendum)
    pred_genus_stems, pred_genus_lemmas = get_gt_stems_lemmas(gt_data, pred_mask_genus)
    #correct
    correct_definiendum_stems, correct_definiendum_lemmas = get_stems_lemmas(gt_data, \
                                correct_mask_definiendum, gt_definiendum_stems, gt_definiendum_lemmas)
    correct_genus_stems, correct_genus_lemmas = get_stems_lemmas(gt_data, \
                                correct_mask_genus, gt_genus_stems, gt_genus_lemmas)
    #false
    false_definiendum_stems, false_definiendum_lemmas = get_gt_stems_lemmas(gt_data, false_mask_definiendum)
    false_genus_stems, false_genus_lemmas = get_gt_stems_lemmas(gt_data, false_mask_genus)

    #undetected
    undetected_definiendum_stems, undetected_definiendum_lemmas = get_stems_lemmas(gt_data, \
                                undetected_mask_definiendum, gt_definiendum_stems, gt_definiendum_lemmas)
    undetected_genus_stems, undetected_genus_lemmas = get_stems_lemmas(gt_data, \
                                undetected_mask_genus, gt_genus_stems, gt_genus_lemmas)

    # print("Before correction:")
    # print(len(gt_definiendum_lemmas+gt_genus_lemmas))
    # print(len(pred_definiendum_lemmas+pred_genus_lemmas))
    # print(len(correct_definiendum_lemmas+correct_genus_lemmas))
    # print(len(false_definiendum_lemmas+false_genus_lemmas))
    # print(len(undetected_definiendum_lemmas+undetected_genus_lemmas))


    # correction
    undetected_definiendum_stems = [w for w in undetected_definiendum_stems if w not in correct_definiendum_stems]
    undetected_definiendum_lemmas = [w for w in undetected_definiendum_lemmas if w not in correct_definiendum_lemmas]
    undetected_genus_stems = [w for w in undetected_genus_stems if w not in correct_genus_stems]
    undetected_genus_lemmas = [w for w in undetected_genus_lemmas if w not in correct_genus_lemmas]

    pred_definiendum_stems = [(s1,w1) for s1,w1 in pred_definiendum_stems for s2,w2 in correct_definiendum_stems \
                                if s1==s2 and (w1 in w2 or w2 in w1)]
    pred_definiendum_lemmas = [(s1,w1) for s1,w1 in pred_definiendum_lemmas for s2,w2 in correct_definiendum_lemmas \
                                if s1==s2 and (w1 in w2 or w2 in w1)]
    pred_genus_stems = [(s1,w1) for s1,w1 in pred_genus_stems for s2,w2 in correct_genus_stems \
                                if s1==s2 and (w1 in w2 or w2 in w1)]
    pred_genus_lemmas = [(s1,w1) for s1,w1 in pred_genus_lemmas for s2,w2 in correct_genus_lemmas \
                                if s1==s2 and (w1 in w2 or w2 in w1)]
                          
    false_definiendum_stems = [(s1,w1) for s1,w1 in false_definiendum_stems for s2,w2 in pred_definiendum_stems \
                            if s1==s2 and (w1 not in w2 and w2 not in w1)]
    false_definiendum_lemmas = [(s1,w1) for s1,w1 in false_definiendum_lemmas for s2,w2 in pred_definiendum_lemmas \
                            if s1==s2 and (w1 not in w2 and w2 not in w1)]
    false_genus_stems = [(s1,w1) for s1,w1 in false_genus_stems for s2,w2 in pred_genus_stems \
                            if s1==s2 and (w1 not in w2 and w2 not in w1)]
    false_genus_lemmas = [(s1,w1) for s1,w1 in false_genus_lemmas for s2,w2 in pred_genus_lemmas \
                            if s1==s2 and (w1 not in w2 and w2 not in w1)]

    correct_definiendum_stems = [(s1,w1) for s1,w1 in pred_definiendum_stems for s2,w2 in correct_definiendum_stems \
                                if s1==s2 and (w1 in w2 or w2 in w1)]
    correct_definiendum_lemmas = [(s1,w1) for s1,w1 in pred_definiendum_lemmas for s2,w2 in correct_definiendum_lemmas \
                                if s1==s2 and (w1 in w2 or w2 in w1)]
    correct_genus_stems = [(s1,w1) for s1,w1 in pred_genus_stems for s2,w2 in correct_genus_stems \
                                if s1==s2 and (w1 in w2 or w2 in w1)]
    correct_genus_lemmas = [(s1,w1) for s1,w1 in pred_genus_lemmas for s2,w2 in correct_genus_lemmas \
                                if s1==s2 and (w1 in w2 or w2 in w1)]
                                
    pred_definiendum_stems += false_definiendum_stems
    pred_definiendum_lemmas += false_definiendum_lemmas
    pred_genus_stems += false_genus_stems
    pred_genus_lemmas += false_genus_lemmas


    # print("After correction:")
    # print(len(set(gt_definiendum_lemmas+gt_genus_lemmas)))
    # print(len(set(pred_definiendum_lemmas+pred_genus_lemmas)))
    # print(len(set(correct_definiendum_lemmas+correct_genus_lemmas)))
    # print(len(set(false_definiendum_lemmas+false_genus_lemmas)))
    # print(len(set(undetected_definiendum_lemmas+undetected_genus_lemmas)))


    pred_edge_list, pred_nodes =  generate_nodes_and_edges_list(pred_definiendum_stems, pred_definiendum_lemmas, \
                                                        pred_genus_stems, pred_genus_lemmas)
    gt_edge_list, gt_nodes =  generate_nodes_and_edges_list(gt_definiendum_stems, gt_definiendum_lemmas, \
                                                        gt_genus_stems, gt_genus_lemmas)
    correct_edge_list, correct_nodes =  generate_nodes_and_edges_list(correct_definiendum_stems, correct_definiendum_lemmas, \
                                                        correct_genus_stems, correct_genus_lemmas)
    false_edge_list, false_nodes =  generate_nodes_and_edges_list(false_definiendum_stems, false_definiendum_lemmas, \
                                                        false_genus_stems, false_genus_lemmas)
    undetected_edge_list, undetected_nodes =  generate_nodes_and_edges_list(undetected_definiendum_stems, undetected_definiendum_lemmas, \
                                                    undetected_genus_stems, undetected_genus_lemmas)
    # print("Number of nodes")
    # print(len(gt_nodes))
    # print(len(pred_nodes))
    # print(len(correct_nodes))
    # print(len(false_nodes))
    # print(len(undetected_nodes))

    G_pred, pred_node_names = generate_graph_with_node_names(pred_edge_list, pred_nodes)
    G_gt, gt_node_names = generate_graph_with_node_names(gt_edge_list, gt_nodes)
    G_correct, correct_node_names = generate_graph_with_node_names(correct_edge_list, correct_nodes)
    G_false, false_node_names = generate_graph_with_node_names(false_edge_list, false_nodes)
    G_undetected, undetected_node_names = generate_graph_with_node_names(undetected_edge_list, undetected_nodes)
    
    plot_graph(G_pred, G_correct, G_false, G_undetected, \
                correct_node_names, false_node_names, undetected_node_names)




####################################################################
#                              main                                #
####################################################################


root = os.getcwd()
experiments_paths = [os.path.join(root,"data/experiments",f) for f in  \
                    os.listdir(os.path.join(root,"data/experiments")) \
                    if os.path.isdir(os.path.join(root,"data/experiments",f)) and "reg" not in f]
mo = -1
for experiment_path in experiments_paths:
    gt_path, pred_path, mo = get_path(experiment_path, mo)
    if pred_path == None:
        mo = -1
        continue
    
    gt_data, pred_data = get_data(gt_path, pred_path)
    if gt_data != None and pred_data != None:
        print("Extracted data...")
        main(gt_data, pred_data)
        print("Graph drawn and can be found in the experiment folder.")
