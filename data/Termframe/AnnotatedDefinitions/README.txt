TermFrame Annotated Definitions
*******************************

The files in .tsv format contain exported annotated definitions that had been manually annotated and curated in the WebAnno tool within the TermFrame project (termframe.ff.uni-lj.si). 


File format:

Header: specifies the annotation layers

#FORMAT=WebAnno TSV 3.2
#T_SP=webanno.custom.Canonicalform|Canonical
#T_SP=webanno.custom.Category|Category
#T_SP=webanno.custom.Definitionelement|Def_element
#T_SP=webanno.custom.Relation|Relation
#T_SP=webanno.custom.Relation_definitor|Rel_verb_frame

Body: 8-column tab-separated format

1- token id
2- position (characters)
3- token
4- canonical form
5- category
6- definition element
7- relation
8- relation frame

#Text=An  aquifer is a body  of rock that can store and transmit significant quantities of water (Gunn, 2004).
1-1	0-2	An	_	_	_	_	_	
1-2	4-11	aquifer	_	C. Geome	DEFINIENDUM	_	_	
1-3	12-14	is	_	_	DEFINITOR[14]	_	_	
1-4	15-16	a	_	_	DEFINITOR[14]	_	_	
1-5	17-21	body	_	_	GENUS[15]	_	_	
1-6	23-25	of	_	_	GENUS[15]	_	_	
1-7	26-30	rock	_	D.1 Abiotic	GENUS[15]	_	_	
1-8	31-35	that	_	_	_	_	_	
1-9	36-39	can	_	_	_	HAS\_FUNCTION[46]	frame\_FUNCTION	

There are 49 .tsv files for English, 46 for Croatian and 53 for Slovene.


For details what each annotation layer means, please refer to 
Vintar, Š., Saksida, A., Vrtovec, K., Stepišnik, U. (2019) Modelling Specialized Knowledge With Conceptual Frames: The TermFrame Approach to a Structured Visual Domain Representation. Proceedings of eLex 2019, https://elex.link/elex2019/wp-content/uploads/2019/09/eLex_2019_17.pdf.

A tool was developed to parse and reformat annotated definitions into a more readable csv format: webanno2csv by Vid Podpečan, available here: https://github.com/vpodpecan/webanno2csv



