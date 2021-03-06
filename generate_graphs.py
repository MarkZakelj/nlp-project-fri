import csv
import os

import networkx as nx
from matplotlib import pyplot as plt
from nltk.stem import PorterStemmer, WordNetLemmatizer

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
import warnings

########################## PARAMETERS ##########################
SECOND_TAG = 'GENUS' #GENUS, HAS_LOCATION, HAS_FORM, HAS_FUNCTION, HAS_SIZE, COMPOSITION, DEFINED_AS
                     #HAS_ATTRIBUTE, HAS_CAUSE, HAS_RESULT, MEASURES, CONTAINS
MAX_WCC_TO_DISPLAY = 3
################################################################

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
        if row1[header.index('Word')] != row2[header.index('Word')] and row2[1] != '[UNK]':
            print("Conflict: words in row {} are not the same:".format(row1[0]))
            print(row1[1]," | ",row2[1])
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

def get_raw_stems_lemmas(data, mask):
    stem = ''
    lemma = ''
    stems = []
    lemmas = []
    for i, is_tag in enumerate(mask):
        if is_tag:
            stem += ' ' + ps.stem(data[i][WORD_COL])
            lemma += ' ' + lemmatizer.lemmatize(data[i][WORD_COL])
            sent_nr = data[i][SENT_COL] # serves as a node number
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
            if len(word) > 2:
                for j, (s, l) in enumerate(gt_lemmas):
                    # correct predictions
                    if word in l and sent_nr == s and \
                    (gt_lemmas[j] not in lemmas or gt_stems[j] not in stems):
                        stems.append(gt_stems[j])
                        lemmas.append(gt_lemmas[j])
                                       
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

def plot_graph(G_pred, G_gt, G_correct, G_false, G_u, \
               correct_node_names, false_node_names, u_node_names, save_dir):
    G = nx.MultiDiGraph()
    G.add_nodes_from(G_pred.nodes(data=True))
    G.add_edges_from([e for e in G_pred.edges])
    G.add_nodes_from(G_u.nodes(data=True))
    G.add_edges_from([e for e in G_gt.edges])


    fig = plt.figure(figsize=(20, 20), frameon=False)
    layout = nx.spring_layout(G, seed=3113794657)

    # options
    options_ = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.6}
    options = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.9}

    # nodes
    nx.draw_networkx_nodes(G, layout, nodelist=G_u.nodes, node_color="tab:gray", **options_) #undetected nodes
    nx.draw_networkx_nodes(G, layout, nodelist=G_false.nodes, node_color="tab:red", **options_) #false nodes
    nx.draw_networkx_nodes(G, layout, nodelist=G_correct.nodes, node_color="tab:green", **options) #correct nodes
    # edges
    nx.draw_networkx_edges(G, layout, width=1, alpha=0.5, arrowsize=11, arrowstyle='->')
    # labels
    nx.draw_networkx_labels(G, layout, u_node_names, font_size=16, font_color="gray")
    nx.draw_networkx_labels(G, layout, false_node_names, font_size=16, font_color="red")
    nx.draw_networkx_labels(G, layout, correct_node_names, font_size=16, font_color="green")
    #fig.suptitle('Complete graph of relations', fontsize=16)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir,"graph.png"))

    # max wcc
    f = plt.figure(figsize=(20, 20),  frameon=False)
    G_ = nx.MultiDiGraph()
    for i, max_wcc in enumerate(sorted(nx.weakly_connected_components(G), key=len, reverse=True)):
        if i in range(MAX_WCC_TO_DISPLAY):
            G_.add_nodes_from(nx.subgraph(G, max_wcc).nodes(data=True))
            G_.add_edges_from([e for e in nx.subgraph(G, max_wcc).edges])
        else:
            break
    
    G_mwcc = nx.subgraph(G, G_)
    try:
        pos = nx.planar_layout(G_mwcc)
    except:
        pos = nx.spring_layout(G_mwcc)
    correct_wcc_names = {}
    false_wcc_names = {}
    undetected_wcc_names = {}
    for n, name in G_mwcc.nodes(data=True):
        if n in correct_node_names:
            correct_wcc_names[n] = name["name"]
        elif n in false_node_names:
            false_wcc_names[n] = name["name"]
        elif name:
            undetected_wcc_names[n] = name["name"]


    # options
    options_ = {"edgecolors": "tab:gray", "node_size": 1000, "alpha": 0.6}
    options = {"edgecolors": "tab:gray", "node_size": 1000, "alpha": 0.8}
    # nodes
    nx.draw_networkx_nodes(G_mwcc, pos, nodelist=[n for n in G_mwcc.nodes if n in G_u.nodes], node_color="tab:gray", **options) #gt nodes
    nx.draw_networkx_nodes(G_mwcc, pos, nodelist=[n for n in G_mwcc.nodes if n in G_false.nodes], node_color="tab:red", **options_) #false nodes
    nx.draw_networkx_nodes(G_mwcc, pos, nodelist=[n for n in G_mwcc.nodes if n in G_correct.nodes], node_color="tab:green", **options) #correct nodes
    # edges
    nx.draw_networkx_edges(G_mwcc, pos, width=7, alpha=0.5, arrowsize=80, arrowstyle='->')
    # labels
    nx.draw_networkx_labels(G_mwcc, pos, undetected_wcc_names, font_size=25, font_color="grey")
    nx.draw_networkx_labels(G_mwcc, pos, false_wcc_names, font_size=25, font_color="red")
    nx.draw_networkx_labels(G_mwcc, pos, correct_wcc_names, font_size=25, font_color="green")
    #fig.suptitle('Relations graph of largest {} connected componens'.format(MAX_WCC_TO_DISPLAY), fontsize=16)
    plt.tight_layout()
    f.savefig(os.path.join(save_dir,"max_wcc_graph.png"))


def preprocess_and_draw(gt_data, pred_data, save_dir):

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
    gt_definiendum_stems, gt_definiendum_lemmas = get_raw_stems_lemmas(gt_data, gt_mask_definiendum)
    gt_genus_stems, gt_genus_lemmas = get_raw_stems_lemmas(gt_data, gt_mask_genus)
    #pred
    pred_definiendum_stems, pred_definiendum_lemmas = get_raw_stems_lemmas(gt_data, pred_mask_definiendum)
    pred_genus_stems, pred_genus_lemmas = get_raw_stems_lemmas(gt_data, pred_mask_genus)
    #correct
    correct_definiendum_stems, correct_definiendum_lemmas = get_stems_lemmas(gt_data, \
                                correct_mask_definiendum, gt_definiendum_stems, gt_definiendum_lemmas)
    correct_genus_stems, correct_genus_lemmas = get_stems_lemmas(gt_data, \
                                correct_mask_genus, gt_genus_stems, gt_genus_lemmas)
    #false
    false_definiendum_stems, false_definiendum_lemmas = get_raw_stems_lemmas(gt_data, false_mask_definiendum)
    false_genus_stems, false_genus_lemmas = get_raw_stems_lemmas(gt_data, false_mask_genus)

    #undetected
    undetected_definiendum_stems, undetected_definiendum_lemmas = get_stems_lemmas(gt_data, \
                                undetected_mask_definiendum, gt_definiendum_stems, gt_definiendum_lemmas)
    undetected_genus_stems, undetected_genus_lemmas = get_stems_lemmas(gt_data, \
                                undetected_mask_genus, gt_genus_stems, gt_genus_lemmas)

    def pad(word):
        return ' '+word+' '

    def correction(stems_1, lemmas_1, stems_2, lemmas_2):
        correct = [(s2,l2) for s1,l1 in zip(stems_1, lemmas_1) \
                        for s2,l2 in zip(stems_2, lemmas_2) \
                        if l1[0]==l2[0] and (l1[1] in l2[1] or l2[1] in l1[1])]
        correct_stems = [g[0] for g in correct]
        correct_lemmas = [g[1] for g in correct]
        return correct_stems, correct_lemmas

    # correction
    undetected_definiendum_stems = [w for w in undetected_definiendum_stems if w not in correct_definiendum_stems]
    undetected_definiendum_lemmas = [w for w in undetected_definiendum_lemmas if w not in correct_definiendum_lemmas]
    undetected_genus_stems = [w for w in undetected_genus_stems if w not in correct_genus_stems]
    undetected_genus_lemmas = [w for w in undetected_genus_lemmas if w not in correct_genus_lemmas]

    false_definiendum_stems = [(s1,w1) for s1,w1 in false_definiendum_stems for s2,w2 in gt_definiendum_stems \
                            if s1==s2 and (pad(w1) not in pad(w2) and pad(w2) not in pad(w1))]
    false_definiendum_lemmas = [(s1,w1) for s1,w1 in false_definiendum_lemmas for s2,w2 in gt_definiendum_lemmas \
                            if s1==s2 and (pad(w1) not in pad(w2) and pad(w2) not in pad(w1))]
    false_genus_stems = [(s1,w1) for s1,w1 in false_genus_stems for s2,w2 in gt_genus_stems \
                            if s1==s2 and (pad(w1) not in pad(w2) and pad(w2) not in pad(w1))]
    false_genus_lemmas = [(s1,w1) for s1,w1 in false_genus_lemmas for s2,w2 in gt_genus_lemmas \
                            if s1==s2 and (pad(w1) not in pad(w2) and pad(w2) not in pad(w1))]

    pred_definiendum_stems, pred_definiendum_lemmas = correction(
                                                    pred_definiendum_stems, pred_definiendum_lemmas,
                                                    gt_definiendum_stems, gt_definiendum_lemmas)
    pred_genus_stems, pred_genus_lemmas = correction(
                                                pred_genus_stems, pred_genus_lemmas,
                                                gt_genus_stems, gt_genus_lemmas)

    correct_definiendum_stems, correct_definiendum_lemmas = correction(
                                                pred_definiendum_stems, pred_definiendum_lemmas,
                                                correct_definiendum_stems, correct_definiendum_lemmas)
    correct_genus_stems, correct_genus_lemmas = correction(
                                                pred_genus_stems, pred_genus_lemmas,
                                                correct_genus_stems, correct_genus_lemmas)

    pred_definiendum_stems += false_definiendum_stems
    pred_definiendum_lemmas += false_definiendum_lemmas
    pred_genus_stems += false_genus_stems
    pred_genus_lemmas += false_genus_lemmas

    gt_definiendum_stems = correct_definiendum_stems + undetected_definiendum_stems
    gt_definiendum_lemmas = correct_definiendum_lemmas + undetected_definiendum_lemmas
    gt_genus_stems = correct_genus_stems + undetected_genus_stems
    gt_genus_lemmas = correct_genus_lemmas + undetected_genus_lemmas

    # nodes and edges 
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

    G_pred, pred_node_names = generate_graph_with_node_names(pred_edge_list, pred_nodes)
    G_gt, gt_node_names = generate_graph_with_node_names(gt_edge_list, gt_nodes)
    G_correct, correct_node_names = generate_graph_with_node_names(correct_edge_list, correct_nodes)
    G_false, false_node_names = generate_graph_with_node_names(false_edge_list, false_nodes)
    G_undetected, undetected_node_names = generate_graph_with_node_names(undetected_edge_list, undetected_nodes)
    
    # drawing
    plot_graph(G_pred, G_gt, G_correct, G_false, G_undetected, \
                correct_node_names, false_node_names, undetected_node_names, save_dir)


def main():
    root = os.getcwd()
    experiments_paths = [os.path.join(root,"data/experiments",f) for f in  \
                        os.listdir(os.path.join(root,"data/experiments")) \
                        if os.path.isdir(os.path.join(root,"data/experiments",f)) and "reg" not in f and 'gen' in f]

    for experiment_path in experiments_paths:
        # FETCH DATA PATHS
        if os.path.exists(os.path.join(experiment_path, "test.csv")):
            gt_path = os.path.join(experiment_path, "test.csv")
        else:
            raise Exception("No test.csv file found, you should run data pre-processing scripts.")    
        model_dirs = [f for f in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path,f))]
        if len(model_dirs)!=0:
            for model in model_dirs:
                if os.path.exists(os.path.join(experiment_path,model,"annotation.csv")):
                    pred_path = os.path.join(experiment_path,model,"annotation.csv")
                else:
                    print("No annotation.csv file found, you should run data pre-processing scripts.")
                    continue
                
                # FETCH DATA AND GENERATE GRAPHS
                save_dir = os.path.join(experiment_path,model)
                print(save_dir)
                gt_data, pred_data = get_data(gt_path, pred_path)
                if gt_data and pred_data:
                    print("Succesfully extracted data, starting drawing...")
                    preprocess_and_draw(gt_data, pred_data, save_dir)
                    print("Graph drawn and can be found in the experiment folder.")
                else:
                    print("Moving to next experiment...")
                print()
                
        else:
            print("No trained models found in:\n{}".format(experiment_path))
            print()
            continue
        


if __name__ == '__main__':
    main()