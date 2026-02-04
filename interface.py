"""
This script provides a convenient interface to tmQM-RDF.

The class TmQMRDFGraph allows to quickly extract the subgraph corresponding to any TMC reported in the dataset.
Access to tmQM-RDF is enabled by initialising the static field TmQMRDFInterface.path_to_tmQM_RDF to the 'graphs' folder.

This script can be run directly to initialise the interface. Type "help(TmQMRDFGraph)" for the documentation.
"""

from pathlib import Path

import urllib.request as url
import rdflib as rdf
import pandas as pd
import numpy as np
import tempfile
import graphviz
import re
import os

import warnings
np.warnings = warnings

# %% tmQM-RDF Interface
class TmQMRDFInterface:
    path_to_tmQM_RDF = None
    
    def __init__(self, tmc_name):
        if self.path_to_tmQM_RDF is None:
            raise Exception("TmQMRDFInterface.path_to_tmQM_RDF not set!")
        
        self.tmc_name = tmc_name
        self._rdf_file = Path(os.path.join(type(self).path_to_tmQM_RDF, f"{tmc_name}.ttl")).absolute()
        self.rdf = rdf.Graph()
        self.rdf.parse(self._rdf_file)
    
    def query(self, query_object):
        """
        Wrapper for self.rdf.query.
        
        See rdflib.query.
        """
        return self.rdf.query(query_object)
        
class TmQMRDFGraph(TmQMRDFInterface):
    """
    A utility class for accessing the tmQM-RDF knwoledge graph.
    
    Upon initialisation with a CSD code, the class will extract the data relative to the TMC and populate the following attributes:
        - rdf: the rdflib.Graph() representation of the TMC
        - CSD_code: the CSD code
        - atoms: a list of pairs, where the first element is the atom URI (minus the prefix) and the second is the chemical symbol
        - bonds: a list of pairs, where each element is the URI (minus the prefix) of an atom involved in the bond
        - ligands: a dictionary indexed by the URIs (minus the prefixes) of the ligands, each value is another dictionary with the following fields:
            - class: the ligand ID (in the tmQMg-L sense)
            - components: a list of the URIs (minus the prefixes) that compose the ligand
        - metal_centre: a dictionary with the same class and components fields as above (the class is just the chemical symbol of the centre)
        
    The class exposes the following methods:
        - query: a wrapper for self.rdf.query
        - as_graphviz: produces a graphical representation of the TMC expressed as graphviz source code.
            This function relies on the PubChem csv version of the periodic table for reference colors of chemical elements.
            The path to the PubChemElements_all.csv file must be provided as the value of the static field TmQMRDFGraph.path_to_chem_info.
            If the file is not available at the specified path, it will be downloaded from 
                https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV?response_type=save&response_basename=PubChemElements_all
        - view: displays the graphviz representation of the TMC and allows it to save it to file
        - render: similar to view, allows to save the representation to a file without visualising it
    """
    
    
    pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV?response_type=save&response_basename=PubChemElements_all"
    path_to_chem_info = None
    
    def __init__(self, tmc_name):
        """
        Initialises the TMC graph from an RDF file
        
        Parameters:
            tmc_name: the CSD code of the desired TMC
        """
        super().__init__(tmc_name)
        
        # Retrieve information such as CSD code, atoms, bonds and ligands/metal centre
        self.CSD_code = tmc_name.replace("-", "_") # Sanitation is needed in case the interface is needed for constructs not coming from the CSD #self._get_CSD_code()
        self.atoms = self._get_atoms()
        self.bonds = self._get_chemical_bonds()
        self.ligands, self.metal_centre = self._get_ligands_components()
    
    def _get_CSD_code(self):
        """
        (Private method)
        
        Runs a SPARQL query to extract the CSD code from the RDF graph
        
        Returns:
            A string containing the CSD code
        """
        
        retrieve_CSD_code = """
            SELECT DISTINCT ?code
            WHERE {
                ?tmc <resource://integreat/p5/complex/TMC/property/meta_data> [
                    <resource://integreat/p5/complex/TMC/CSD_code> ?code
                ] .    
            }
            """
            
        qres = self.rdf.query(retrieve_CSD_code)
        
        return [str(row.code) for row in qres][0]
    
    def _get_atoms(self):
        """
        (Private method)
        
        Runs a SPARQL query to extract the atoms of the TMC from the RDF graph
        
        Returns:
            A list of pairs (atom_name, atom_element), where
                - atom_name is the name of the object representing the atom, intended
                    as the last element on the URI path (e.g. if the URI is resource://integreat/p5/atomic/atom/XXYYZZ_El1,
                    the name is XXYYZZ_El1)
                - atom_element is the chemical symbol of the element of the atom, again, extracted as the last element
                    of the URI path of the element object
        """
        
        retrieve_atoms = """
            SELECT DISTINCT ?atom ?element
            WHERE {
                ?atom <resource://integreat/p5/atomic/atom/isAtom> ?element .
            }
            """
            
        qres = self.rdf.query(retrieve_atoms)
        
        return [(str(row.atom).split('/')[-1], str(row.element).split('/')[-1]) for row in qres]
    
    def _get_chemical_bonds(self):
        """
        (Private method)
        
        Runs a SPARQL query to extract the chemical bonds within the TMC from the RDF graph.
        
        Returns:
            A list of pairs (atom1_name, atom2_name), where atom[n]_name is the name of the atom at the
                n-th end of the bond (n = 1,2). The atoms are ordered according to the IDs assigned in the
                tmQMg dataset, with id(atom1) < id(atom2).
        """
        
        retrieve_bonds = """
            SELECT DISTINCT ?first ?second
            WHERE {
                   ?first <resource://integreat/p5/atomic/structure/b> ?bnd .
                   ?second <resource://integreat/p5/atomic/structure/b> ?bnd .
                   
                   FILTER (?first != ?second)
            }
            """
        
        qres = self.rdf.query(retrieve_bonds)
        
        all_bonds = [(str(row.first).split('/')[-1], str(row.second).split('/')[-1]) for row in qres]
        
        bonds = []
        # Remove duplicates
        for bond in all_bonds:
            if (bond[1], bond[0]) not in bonds:
                bonds += [bond]
        
        return bonds
    
    def _get_ligands_components(self):
        """
        (Private method)
        
        Runs a SPARQL query to extract the ligands (and the metal centre) of the TMC from the RDF graph.
        Notice that the metal centre is structurally represented as a ligand for simplicity.
        
        Returns:
            A pair of two dictionaries:
                - {ligand_name: {'class': ligand_id, 'components': [...]}, ...}
                    where:
                        - ligand_name is the name of the ligand object (intended as the last element
                          on the URI path)
                        - ligand_id is the id of the reference ligand
                        - components is a list of the names of the atoms that compose this instance of the ligand
                - {'class': centre_element, 'components': [centre_atom]}
                    where:
                        - centre_element is the chemical element of the metal centre
                        - centre_atom is the name of the atom representing the centre at the atomic level
        """
        
        retrieve_ligands = """
            SELECT DISTINCT ?ligand ?atom ?ligand_type
            WHERE {
                ?ligand <resource://integreat/p5/ligand/ligand/hasAtom> ?atom .   
                ?ligand <resource://integreat/p5/ligand/ligand/isLigand> ?ligand_type .
            }
            """
            
        qres = self.rdf.query(retrieve_ligands)
        
        out_lig = {}
        
        for row in qres:
            ligand = str(row.ligand).split("/")[-1]
            atom = str(row.atom).split("/")[-1]
            ligand_type = str(row.ligand_type).split("/")[-1].split("_")[-1]
            
            current_comp_list = out_lig.get(ligand, {"components": []})["components"]
            
            new_comp_list = current_comp_list + [atom]
            
            out_lig[ligand] = {"class": ligand_type, "components": new_comp_list}
        
        retrieve_centre = """
            SELECT DISTINCT ?metal_centre ?atom ?metal_centre_type
            WHERE {
                ?metal_centre <resource://integreat/p5/ligand/centre/hasAtom> ?atom .   
                ?metal_centre <resource://integreat/p5/ligand/centre/isMetalCentre> ?metal_centre_type .
            }
            """
            
        qres = self.rdf.query(retrieve_centre)
        
        for row in qres:
            out_centre = {
                "class": str(row.metal_centre_type).split("/")[-1].split("_")[-1],
                "components": [str(row.atom).split("/")[-1]]
            }
        
        return out_lig, out_centre
    
    def as_graphviz(self, layout = "neato"):
        """
        Encodes the TMC as a graph described via the DOT language.
        In the resulting graph, each ligand (and the metal centre) is represented as a cluster and hence highlighted
        via a box named after the ligand id, so as to emphasize the structure of the ligand level. The entire graph
        is a cluster itself, so it is also highlighted by a box, representing the complex level. The atoms are labelled
        and colored according to their chemcial element, using the information extracted from the PubChem periodic table data
        downloaded from https://pubchem.ncbi.nlm.nih.gov/periodic-table/#view=list .
        
        Parameters:
            layout: the desired layout engine, one of 'dot' and 'neato'.
            
        Returns:
            A graphiv.Source object in which the TMC has been encoded
        """
        
        if self.path_to_chem_info is None:
            raise Exception("TmQMRDFGraph.path_to_chem_info not set!")
        
        # Preprocess node statements
        #
        # For each atom, a node statement is created. This statement can then be accessed via the node name
        
        node_statements = {}
        
        for node, _ in self.atoms:
            node_statements[node] = NodeStatement(node)
        
        # Preprocess edge statements
        #
        # For each edge, an edge statement is created. This statement can be accessed via the pair (atom1_name, atom2_name).
        # See self._get_chemical_bonds for more information about the atom names.
        
        edge_statements = {}
        
        for at1, at2 in self.bonds:
            edge_statements[at1, at2] = EdgeStatement((at1, at2))
        
        
        # Define the graph object
        #
        # Create a graph and initialize the graph-level aesthetic properties (font and rankdir)
        #
        # Then add the graph-level attributes for nodes and edges
        
        G = Graph(
                name = self.CSD_code,
                fontname = "Helvetica,Arial,sans-serif",
                rankdir = "LR",
                compound = "true",
                layout = layout
            )
        
        G.add_statements(NodeStatement(
                fontname = "Helvetica,Arial,sans-serif",
                penwidth = 1.0, 
                color = "black"
                # width = 0.5,
                # height = 0.5
            ))
        
        G.add_statements(EdgeStatement(
                fontname = "Helvetica,Arial,sans-serif",
                len = 0.6
            ))
            
        
        # Create elemental subgraphs
        #
        # For each element encountered in the TMC, create a subgraph which collects all the atoms which possess
        # that element, and then for each subgraph define the attributes that apply a unique style to every atom
        # of the given element
        
        # Read PubChem information
        if not os.path.exists(os.path.join(type(self).path_to_chem_info, "PubChemElements_all.csv")):
            
            if not os.path.exists(type(self).path_to_chem_info):
                os.makedirs(type(self).path_to_chem_info)
                
            url.urlretrieve(
                    type(self).pubchem_url,
                    os.path.join(type(self).path_to_chem_info, "PubChemElements_all.csv")
                )
        
        pubchem = pd.read_csv(os.path.join(type(self).path_to_chem_info, "PubChemElements_all.csv"), sep = ",")
        
        # Extract the set of the elements found in the TMC
        elements = set([el for _, el in self.atoms])
        
        # Then, for each element...
        subgraphs = []
        for el in elements:
            # Extract all the node statements which pertain to the element in question (they will be
            #   added to the subgraph to assert the belonging of an atom)
            statements_by_el = [node_statements[node] for node, el0 in self.atoms if el0 == el]
            
            # Extract the PubChem information relative to the given element
            el_name = pubchem.loc[pubchem["Symbol"] == el, "Name"].iloc[0]
            el_col = pubchem.loc[pubchem["Symbol"] == el, "CPKHexColor"].iloc[0]
            
            # Check that the color extracted from PubChem is a valid HEX color, using a regex
            #   If the color is not valid, use a default color (pink)
            if not re.match(r"^(?:[0-9a-fA-F]{2}){3}$", el_col):
                el_col = "f78bb2"
            
            # Set up a font color: white for carbon (because of the dark background), black for all the others
            font_col = "black"
            if el == "C":
                font_col = "white"
            
            # Define the node statement that will apply the aesthetic attributes to all nodes in the subgraph
            node_size = "0.3" if el == "H" else "0.5"
            fontsize = "7" if el == "H" else "14"
            header = NodeStatement(
                style = "filled",
                fillcolor = f"#{el_col}",
                label = el,
                fontcolor = font_col,
                width = node_size,
                height = node_size,
                fontsie = fontsize,
                fixedsize = True
            )
            
            # Create the subgraph and save it
            subgraphs += [
                    Subgraph(
                            header,
                            *statements_by_el,
                            name = el_name
                        )
                ]
        
        # add all the subgraphs to the graph object
        G.add_statements(*subgraphs)
        
        # Create metal centre cluster (structurally speaking, same as a ligand)
        
        centre_name = self.metal_centre["components"][0]
        centre_cluster_name = "metal_centre_" + self.metal_centre["class"]
        
        node_statements[centre_name].attributes["peripheries"] = 3
        ligands_clusters = [
                Cluster(
                        node_statements[centre_name],
                        name = centre_cluster_name,
                        label = f"Metal centre: {self.metal_centre['class']}",
                        color = "blue",
                        fontcolor = "blue"
                    )
            ]
        
        
        # Identify bonds from centre to ligands and atoms bonded to the centre
        #   It will be needed also when creating the ligand clusters
        
        centre_lig_bonds = []
        centre_bonded_atoms = []
        
        # For each bond, isolate those that involve the metal centre
        #   Also, identify if the metal centre is the first (tail) or the second (head)
        #   element of the bond and specify ltail/lhead accordingly
        for edge, edge_statement in edge_statements.items():
            if centre_name in edge:
                edge_statement.attributes[
                        "ltail" if edge[0] == centre_name else "lhead"
                    ] = "cluster_" + centre_cluster_name
                edge_statement.attributes["len"] = 3
                edge_statement.attributes["style"] = "dashed"
                
                centre_lig_bonds += [edge_statement]
                
                centre_bonded_atoms += [edge[1] if edge[0] == centre_name else edge[0]]
        
        # For each ligand - centre bond, highlight ligand binding atoms
        for at in centre_bonded_atoms:
            node_statements[at].attributes["peripheries"] = 2
        
        # Create ligand clusters
        #
        # For each ligand, define a cluster that will encompass all the atoms and all the chenical bonds
        # that belong to the ligand. A bond is said to belong to the ligand when both its ends belong to the ligand
        
        # See self._get_ligands_components docstring for a description of ligand
        #   For each ligand...
        for ligand_name, ligand in self.ligands.items():
            # Extract all the node statements associated with the ligand
            #   except for those bonded to the metal centre (those will be added separately)
            nodes_in_ligand = [node_statements[at] for at in ligand["components"] if at not in centre_bonded_atoms]
            
            # Identify the atoms bonded to the centre and put them in their own subgraph
            nil_bonded_to_centre = [node_statements[at] for at in ligand["components"] if at in centre_bonded_atoms]
            
            btc_subgraph = Subgraph(
                    *nil_bonded_to_centre,
                    rank = "same"
                )
            
            # Extract all the edge statements that belong to the ligand
            edges_in_ligand = []
            
            for edge, edge_statement in edge_statements.items():
                if edge[0] in ligand["components"] and edge[1] in ligand["components"]:
                    edges_in_ligand += [edge_statement]
            
            # Create the cluster using the nodes and the edges and label it according
            #   to the ligand id
            ligands_clusters += [
                    Cluster(
                            *nodes_in_ligand,
                            *edges_in_ligand,
                            btc_subgraph,
                            name = ligand_name.replace("-", "__"),
                            label = ligand["class"],
                            color = "blue",
                            fontcolor = "blue"
                        )
                ]
    
        # Create TMC cluster using all the ligands clusters and the bonds to the metal centre
        
        tmc_cluster = Cluster(
                *centre_lig_bonds,
                *ligands_clusters,
                name = self.CSD_code,
                label = self.CSD_code,
                color = "orange",
                fontcolor = "orange"
            )
        
        
        # Add TMC cluster to the graph object
        
        G.add_statements(tmc_cluster)
        
        return G.assemble()["statement"]
    
    def view(self, format = "png", filename = None, layout = "neato"):
        """
        Produces an image of the TMC rendered via the graphviz module and visualises it.
        
        Parameters:
            - format: the desired output format for the resulting graphviz object (pdf, png, svg, ...)
            - filename: the name of the file (without the extension) to which the output should be saved (optional)
            - layout: the desired layout engine, one of 'dot' and 'neato'.
        """
        src = self.as_graphviz(layout)
        src = graphviz.Source(src, format = format)
        
        if filename is None:
            src.view(tempfile.mktemp(".gv"), cleanup = True)
        else:
            src.view(filename, cleanup = True)
            
    def render(self, format = "png", filename = None, layout = "neato"):
        """
        Produces an image of the TMC rendered via the graphviz module
        
        Parameters:
            - format: the desired output format for the resulting graphviz object (pdf, png, svg, ...)
            - filename: the name of the file (without the extension) to which the output should be saved (optional)
            - layout: the desired layout engine, one of 'dot' and 'neato'.
        """
        src = self.as_graphviz(layout)
        src = graphviz.Source(src, format = format)
        
        if filename is None:
            src.render(tempfile.mktemp(".gv"), cleanup = True)
        else:
            src.render(filename, cleanup = True)

# %% Graphviz utilities
"""
This part of the script implements a series of wrapper classes for the syntax
of the Graphviz language.

The dependency tree of the classes is:
    
    GraphvizStatement
     |
     |-> SimpleStatement
     |    |
     |    |-> NodeStatement *
     |    |
     |    |-> EdgeStatement *
     |    
     |-> ComplexStatement
          |
          |-> GraphStatement
          |    |
          |    |-> Graph *
          |    |
          |    |-> Digraph *
          |
          |-> Subgraph *
               |
               |-> Cluster *
               
The classes marked with * are those meant to be directly employed by the user.


In this script, the templates of the available statements are described using the following syntax:
    
    template_id : template_description
    
Inside the template description, the following rules apply:
    - Other template ids are always written as plain words.
    - User imput is identified by enclosing stars (*).
        E.g.: the token *id* stands for any user-defined id
    - Literal values are enclosed in single quotes (').
        E.g.: the token '=' is to be replaced exactky by the symbol =
    - Alternatives are sparated by vertical bats (|).
        E.g.: the token token_1 | token_2 can be replaced wither by token_1 or by token_2
    - Optional values are enclosed in square brackets ([ ]).
        E.g.: the token token_1 [token_2] is to be replaced either by token_1 or by token_1 token_2
"""

class GraphvizStatement:
    """
    Baseline class

    It simply stores the parameters and exposes the assemble method
    """
    
    def __init__(self, keyword, **kwargs):
        """
        Parameters:
            keyword: the main command that defines the statement (e.g., node, edge, graph, ...)
            **kwargs: a series of key-value pairs that corespond to the attributes allowed by Graphviz for the
                specified statement (optional)
        """
        
        self.keyword = keyword
        self.attributes = kwargs
     
    def assemble(self):
        """
        Placeholder. Exposure of the assemble method.
        
        Assembles the statement by combining together the provided input parameters
        """
        
        pass

class SimpleStatement(GraphvizStatement):
    """
    This class wraps the concept of a Graphviz statement that can be expressed in one line
    using the template

        SimpleStatement         :   keyword [ attribute_list ] ';'
        
        keyword                 :   attribute_statement | declaration_statement
        attribute_statement     :   'node' | 'edge'
        declaration_statement   :   node_declaration | edge_declaration
        node_declaration        :   *node_id*
        edge_declaration        :   *node1_id* edge_type *node2_id*
        edge_type               :   '--' | '->'
        attribute_list          :   '[' list ']'
        list                    :   *key* '=' '"'*value*'"' [',' list]
    """
    
    def __init__(self, keyword, is_attribute, **kwargs):
        """
        Initialises the simple statement.
        
        Parameters:
            keyword: the keyword (see class description), string
            is_attribute: whether this is an attribute statement (see class description) (boolean)
            **kwargs: a series of named arguments defining the attribute list for nodes (string) (optional)
            
        """
        
        super().__init__(keyword, **kwargs)
        
        self.is_attribute = is_attribute
    
    def assemble(self):
        """
        Assembles the statement
        
        Returns:
            a string that follows the SimpleStatement pattern (see class description)
        """
        
        statement = f"{self.keyword}"
        
        if len(self.attributes) > 0:
            statement += " ["
            statement += ", ".join([f"{key} = \"{value}\"" for key, value in self.attributes.items()])
            statement += "]"
        
        statement += ";"
        
        return statement

class NodeStatement(SimpleStatement):
    """
    A class that wraps the concept of a SimpleStatement related to nodes:
        
        NodeStatement   :   keyword [ attribute_list ] ';'
        
        keyword         :   'node' | *node_id*
        attribute_list  :   '[' list ']'
        list            :   *key* '=' '"'*value*'"' [',' list]
    """
    
    def __init__(self, node_id = None, **kwargs):
        """
        Parameters:
            node_id: the id of the node, if None the keyword defaults to 'node' and 
                the statement defaults to an attribute statement (see SimpleStatement 
                class description) (string). Default: None
            **kwargs: a series of named arguments defining the attribute list for nodes (string) (optional)
        """
        
        super().__init__(
            node_id if node_id is not None else "node",
            node_id is None,
            **kwargs
        )

class EdgeStatement(SimpleStatement):
    """
    A class that wraps the concept of a SimpleStatement related to edges:
        
        NodeStatement       :   keyword [ attribute_list ] ';'
        
        keyword             :   'edge' | edge_declaration
        edge_declaration    :   *node1_id* edge_type *node2_id*
        edge_type           :   '--' | '->'
        attribute_list      :   '[' list ']'
        list                :   *key* '=' '"'*value*'"' [',' list]
        
    NOTE: the edge type is automatically determined by the GraphStatement this statement is added to,
    so until this statement is assembled within a graph, the edge type will be replaced by the placeholder %s
    """
    
    def __init__(self, edge = None, **kwargs):
        """
        Parameters:
            edge: a list (or tuple) containing node1_id and node2_id, if None the keyword defaults 
                to 'edge' and the statement defaults to an attribute statement (see SimpleStatement 
                class description). Default: None
            **kwargs: a series of named arguments defining the attribute list for nodes (string) (optional)
        """
        
        super().__init__(
            f"{edge[0]} %s {edge[1]}" if edge is not None else "edge",
            edge is None,
            **kwargs
        )
        
class ComplexStatement(GraphvizStatement):
    """
    This class wraps the concept of a Graphviz statement that may require multiple lines to be expressed,
    using the template

        ComplexStatement    :   keyword [*ID*] '{' statement_list '}'
        
        keyword             :   'graph' | 'digraph' | 'subgraph'
        statement_list      :   statement [statement_list]
        statement           :   attribute | SimpleStatement | ComplexStatement
        attribute           :   *key* '=' '"'*value*'"' ';'
    """
    
    def __init__(self, keyword, *args, **kwargs):
        """
        Initialises the complex statement
        
        Parameters:
            - keyword: the keyword
            *args: a series of SimpleStatement, Subgraph or Cluster instances to be embedded into
                the complex statement (optional)
            **kwargs: a series of named arguments defining the attribute list 
                for graphs, digraphs, subgraphs or clusters (string). If you wish
                to provide an id for the complex statement you have to pass it as a
                parameter named 'name' (to avoid conflict with the 'id' attribute
                allowed by Graphviz) (optional)
        """
        
        super().__init__(keyword, **kwargs)
        
        self.args = args
        
        # The id is not an attribute in the sense of Graphviz syntax, so it
        #   has to be removed
        self.name = self.attributes.pop("name", None)
        
        # Default value of safety flag, classes that do not allow
        #   to be embedded have to change this
        self.can_be_child = True
        
        # Default value of edge type, classes that prescribe a
        #   specific type have to change this
        self.edge_type = "%s"
        
        # Helper variables that will store and classify
        #   the statements other than the attributes
        self.simple_statements = {
                "attribute": {"nodes": [], "edges": []},
                "nonattribute": {"nodes": [], "edges": []}
            }
        self.complex_statements = []
        
        # Add the statements that are not attributes
        self.add_statements(*args)
    
    def add_statements(self, *args):
        """
        Add new SimpleStatement, Subgraph or Cluster instances to be embedded into the complex statement
        
        Parameters:
            *args: a series of SimpleStatement, Subgraph or Cluster instances to be embedded into
                the complex statement (optional)
        """
        
        # Helper variables: allow easy access to the key that correctly stores
        #   the statements in self.simple_statements, depending on whether the simple
        #   statement is an attribute statement and it refers to nodes or edges
        is_attribute = {True: "attribute", False: "nonattribute"}
        is_node = {True: "nodes", False: "edges"}
        
        for statement in args:
            if isinstance(statement, SimpleStatement):
                self.simple_statements[
                        is_attribute[statement.is_attribute]
                    ][
                        is_node[isinstance(statement, NodeStatement)]
                    ]+= [statement]
            else:
                self.complex_statements += [statement]
        
        # Sanity check: verify that all the provided statements are of the correct type
        for stmnt in self.complex_statements:
            if not isinstance(stmnt, ComplexStatement):
                raise Exception(f"Error while adding statements to complex statement: trying to add something that is not a SimpleStatement or a ComplexStatement! Trying to add: {type(stmnt)}")
            
        if not all( stmnt.can_be_child for stmnt in self.complex_statements ):
            raise Exception("Error while adding statements to complex statement: trying to add graph/digraph as a child!")
                
    def assemble(self):
        """
        Assembles the complex statement
        
        Returns:
            A dictionary with two entries:
                - 'statement': A statement following the ComplexStatement template (see class description)
                - 'lines': the statement broke into lines
        """
        
        # Helper variables: the complex statement is written in the order
        #
        #   1. attributes
        #   2. node attribute statements
        #   3. edge attribute statemens
        #   4. complex statements
        #   5. node statements
        #   6. edge statements
        #
        #   and in between each group of statements there should be an empty line, hence it is
        #   necessary to know whether or not each group is empty or has any statements
        #   To ensure a nice formatting, after attempting to write each group, it is necessary
        #   to check if 1. something has been written AND 2. something will be written after. If so
        #   an empty line has to be added. The following variables perform all of these checks beforehand
        have_to_write_attr = len(self.attributes.items()) > 0
        have_to_write_simple_attr_statements = {
                "edges": len(self.simple_statements["attribute"]["edges"]) > 0,
                "nodes": len(self.simple_statements["attribute"]["nodes"]) > 0
            }
        have_to_write_complex_statements = len(self.complex_statements) > 0
        have_to_write_simple_nonattr_statements = {
                "edges": len(self.simple_statements["nonattribute"]["edges"]) > 0,
                "nodes": len(self.simple_statements["nonattribute"]["nodes"]) > 0
            }
        
        # Container list for all the lines of the statement
        statements = []
        
        # Opening line of the statement: keyword [*id*] '{'
        statements += [
                self.keyword \
                + (f" {self.name}" if self.name is not None else "") \
                + " {"
            ]
            
        # Add attributes
        for key, value in self.attributes.items():
            statements += [f'\t{key} = \"{value}\";']
            
        if have_to_write_attr and have_to_write_simple_attr_statements:
            statements += [""] # newline
            
        # Add simple attribute statements (nodes)
        for statement in self.simple_statements["attribute"]["nodes"]:
            statements += ["\t" + statement.assemble()]
        
        if have_to_write_simple_attr_statements["nodes"] and (
                    have_to_write_simple_attr_statements["edges"] or
                    have_to_write_complex_statements or
                    have_to_write_simple_nonattr_statements["nodes"] or
                    have_to_write_simple_nonattr_statements["edges"]
                ):
            statements += [""] # newline
        
        # Add simple attribute statements (edges)
        for statement in self.simple_statements["attribute"]["edges"]:
            statements += ["\t" + statement.assemble()]
            
        if have_to_write_simple_attr_statements["edges"] and (
                    have_to_write_complex_statements or
                    have_to_write_simple_nonattr_statements["nodes"] or
                    have_to_write_simple_nonattr_statements["edges"]
                ):
            statements += [""] # newline
            
        # Add complex statements
        for s_idx in range(len(self.complex_statements)):
            statement = self.complex_statements[s_idx]
            
            # Inherit edge type from enclosing complex statement
            #   Only top objects like graphs and digraphs can
            #   define an edge type
            old_edge_type = statement.edge_type
            statement.edge_type = self.edge_type
            
            # Retrieve child statments and add them to the complex statement lines
            statements_of_child = statement.assemble()["lines"]
            statements += ["\t" + soc for soc in statements_of_child]
            
            # Restore original edge type (%s)
            statement.edge_type = old_edge_type
            
            if s_idx < len(self.complex_statements) - 1:
                statements += [""] # newline
        
        if have_to_write_complex_statements and (
                    have_to_write_simple_nonattr_statements["nodes"] or
                    have_to_write_simple_nonattr_statements["edges"]
                ):
            statements += [""] # newline
            
        # Add simple nonattribute statements (nodes)
        for statement in self.simple_statements["nonattribute"]["nodes"]:
            statements += ["\t" + statement.assemble()]
        
        if have_to_write_simple_nonattr_statements["nodes"] and have_to_write_simple_nonattr_statements["edges"]:
            statements += [""] # newline
        
        # Add simple nonattribute statements (edges)
        for statement in self.simple_statements["nonattribute"]["edges"]:
            statements += ["\t" + (statement.assemble() % self.edge_type)]
        
        # Close statement
        statements += ["}"]
        
        # Return assembled statements
        return {
                "statement": "\n".join(statements), 
                "lines": statements
            }

class GraphStatement(ComplexStatement):
    """
    This clas wraps a ComplexStatement whose keyword is
    either graph or digraph
    """
    
    def __init__(self, keyword, *args, **kwargs):
        """
        Initialises the graph statement
        
        Parameters:
            - keyword: the keyword
            *args: a series of SimpleStatement, Subgraph or Cluster instances to be embedded into
                the graph statement (optional)
            **kwargs: a series of named arguments defining the attribute list 
                for graphs or digraphs (string). If you wish to provide an id for
                the graph statement you have to pass it as a parameter named 'name' 
                (to avoid conflict with the 'id' attribute allowed by Graphviz) (optional)
        """
        
        super().__init__(keyword, *args, **kwargs)
        
        self.can_be_child = False

class Graph(GraphStatement):
    """
    This clas wraps a GraphStatement whose keyword is graph (undirected graph)
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialises the graph statement
        
        Parameters:
            *args: a series of SimpleStatement, Subgraph or Cluster instances to be embedded into
                the graph statement (optional)
            **kwargs: a series of named arguments defining the attribute list 
                for graphs (string). If you wish to provide an id for
                the graph statement you have to pass it as a parameter named 'name' 
                (to avoid conflict with the 'id' attribute allowed by Graphviz) (optional)
        """
        
        super().__init__("graph", *args, **kwargs)
        
        self.edge_type = "--"

class Digraph(GraphStatement):
    """
    This clas wraps a GraphStatement whose keyword is digraph (directed graph)
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialises the digraph statement
        
        Parameters:
            *args: a series of SimpleStatement, Subgraph or Cluster instances to be embedded into
                the digraph statement (optional)
            **kwargs: a series of named arguments defining the attribute list 
                for digraphs (string). If you wish to provide an id for
                the digraph statement you have to pass it as a parameter named 'name' 
                (to avoid conflict with the 'id' attribute allowed by Graphviz) (optional)
        """
        
        super().__init__("digraph", *args, **kwargs)
        
        self.edge_type = "->"

class Subgraph(ComplexStatement):
    """
    This clas wraps a ComplexStatement whose keyword is subgraph
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialises the subgraph statement
        
        Parameters:
            *args: a series of SimpleStatement, Subgraph or Cluster instances to be embedded into
                the digraph statement (optional)
            **kwargs: a series of named arguments defining the attribute list 
                for digraphs (string). If you wish to provide an id for
                the digraph statement you have to pass it as a parameter named 'name' 
                (to avoid conflict with the 'id' attribute allowed by Graphviz) (optional)
        """
        
        super().__init__("subgraph", *args, **kwargs)

class Cluster(Subgraph):
    """
    This clas wraps a ComplexStatement whose keyword is subgraph and which defines a cluster.
    Notice that, in Graphviz, clusters are subgrapphs whose ID starts with 'cluster'
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialises the cluster statement
        
        Parameters:
            *args: a series of SimpleStatement, Subgraph or Cluster instances to be embedded into
                the digraph statement (optional)
            **kwargs: a series of named arguments defining the attribute list 
                for digraphs (string). An ID must be provided by pasing it as a parameter
                named 'name' (to avoid conflict with the 'id' attribute allowed by Graphviz) (optional)
        """
        
        super().__init__(*args, **kwargs)
        
        if self.name is None:
            raise Exception("Error in Cluster initialisation: a cluster must have an id! Provide one by passing the name parameter to the constructor.")
            
        self.name = "cluster_" + self.name

# %% Main statement (for local management)
if __name__ == "__main__":
    TmQMRDFGraph.path_to_tmQM_RDF = os.path.join(".", "graphs")
    TmQMRDFGraph.path_to_chem_info = os.path.join(".")