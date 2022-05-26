# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:22:28 2022

@author: Kert PC
"""

from nltk.stem import PorterStemmer, WordNetLemmatizer
import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


scnd_tag = 'GENUS'

edge_list = []

num_of_nodes = 0
node_hash = {}
node_names = {}

for i, result_sentence in enumerate(tags) : #predictions / GT
    mask_dfd = [tag == 'DEFINIENDUM' for tag in result_sentence]
    mask_gen = [tag == scnd_tag for tag in result_sentence]
    
    definiendums = []
    definiendums_wh = []
    geni = []
    geni_wh = []
    
    defini = ''
    defini_wh = ''
    for j, is_dfd in enumerate(mask_dfd) :
        if is_dfd :
            defini += ' ' + ps.stem(tokens[i][j]) # Iz besed
            defini_wh += ' ' + lemmatizer.lemmatize(tokens[i][j])
        elif defini != '' :
            definiendums.append(defini.strip().lower())
            definiendums_wh.append(defini_wh.strip().lower())
            defini_wh = ''
            defini = ''
            
    genus = ''
    genus_wh = ''
    for j, is_gen in enumerate(mask_gen) :
        if is_gen :
            genus += ' ' + ps.stem(tokens[i][j])
            genus_wh += ' ' + lemmatizer.lemmatize(tokens[i][j])
        elif genus != '' :
            geni.append(genus.strip().lower())
            geni_wh.append(genus_wh.strip().lower())
            genus_wh = ''
            genus = ''
    
    for j, defin in enumerate(definiendums):
        if defin not in node_hash :
            node_hash[defin] = num_of_nodes
            node_names[num_of_nodes] = definiendums_wh[j]
            num_of_nodes += 1
        
        for k, gen in enumerate(geni) :
            if gen not in node_hash :
                node_hash[gen] = num_of_nodes
                node_names[num_of_nodes] = geni_wh[j]
                num_of_nodes += 1
            
            edge_list.append((node_hash[defin], node_hash[gen]))
    
"""
node_name_hash = {}
for node in node_hash :
    node_name_hash[node_hash[node]] = node
"""
G = nx.DiGraph()

for node in node_hash:
    G.add_node(node_hash[node], label=node)
    
for edge in edge_list:
    G.add_edge(edge[0], edge[1])
    
comp_gen = nx.weakly_connected_components(G)
components = [c for c in comp_gen]

col = sns.color_palette("hls", len(components))
colors = [(0,0,0)] * len(node_hash)

for cl, comp in enumerate(components) :
    for node in list(comp) :
        colors[node] = col[cl]

    
fig = plt.figure(figsize=(20, 20))
layout = nx.spring_layout(G)
nx.draw_networkx_nodes(G, layout, node_color=colors)
nx.draw_networkx_edges(G, layout)
nx.draw_networkx_labels(G, layout, node_names)
print('drawn')