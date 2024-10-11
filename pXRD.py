# Copyright 2024 CNPEM
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os
from PyQt5.QtCore import (Qt, pyqtSignal, QRect, QEvent)
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox, QTabWidget,QMainWindow, QPushButton, QVBoxLayout, 
                                QHBoxLayout, QWidget, QSlider, QLabel, QLineEdit, QColorDialog, QCheckBox, QFileDialog)
import qtawesome as qta
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import ToolbarQt
from matplotlib.backend_managers import ToolManager
from matplotlib import backend_tools
from matplotlib.patches import Arrow
from matplotlib.colors import rgb2hex

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['toolbar'] = 'toolmanager'
from matplotlib.backend_tools import ToolBase, ToolToggleBase
import numpy as np
import xrayutilities as xu
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import colorsys, random, inspect

class Language():
    """ This class is used to give information to the users in their language. There are three options: ['en', 'es', 'br'], for English, Spanish and Brazilian Portuguese. 
        If the user want to change, it's necessary to necessary to open the '.default' file and change the language to the desired one, and restart the application"""
    def __init__(self):
        self.names = {'br':{'title':'Padr\u00e3o de Difra\u00e7\u00e3o de Raios X',
                'pars': u'Par\u00e2metros de Rede (\u212b)', 
                'wvl': 'Comprimento de Onda (\u212b) e Energia (keV)',
                'size': 'Tamanho Te\u00f3rico de Cristalito (\u212b)', 
                'hkl': '\u00cdndices de Miller (hkl)', 
                'int': 'Intensidade M\u00e1xima', 
                'atm': '\u00c1tomo', 
                'crystal': 'C\u00e9lula Unit\u00e1ria', 
                'base': '\u00c1tomos da Base', 
                'label_0': '\u00c1tomo da posi\u00e7\u00e3o (x,y,z) = (0,0,0)',
                'info': 'Infos', 
                'welcome': 'Bem vindo ao XRD Playground', 
                'xpd':'Padr\u00e3o de Difra\u00e7\u00e3o de Raios X', 
                'lattpar': 'Par\u00e2metros da Rede',
                'a': 'Par\u00e2metro "a"',  'b': 'Par\u00e2metro "b"', 'c': 'Par\u00e2metro "c"', 
                'alpha': '\u00e2ngulo entre os eixos definidos por "b" e "c"', 'beta': '\u00e2ngulo entre os eixos definidos por "a" e "c"', 'gamma': '\u00e2ngulo entre os eixos definidos por "a" e "b"', 
                'complement_0': 'em \u212b', 
                'complement_1': 'em graus', 
                'problem_a': 'n\u00e3o \u00e9 um valor v\u00e1lido \npara',
                'en_0': 'Energia', 'wvl_0': 'Comprimento de onda', 'size_0': 'Tamanho de cristalito', 'atm_0': 'Posi\u00e7\u00e3o do \u00e1tomo',
                'freeze0':'Congelar curva', 'freeze1':r'cor {}', 'red': 'vermelha', 'green':'verde', 'blue':'azul',
                'rescale':'Ajustar a escala da intensidade',
                'tt_b_pars':'Clique para voltar ao valor inicial', 'tt_s_pars':'Clique e arraste para mudar','tt_e_pars':'Entre um valor',
                'color_0':'Mudar a cor deste \u00e1tomo',
                'crysedge': 'Mostrar/esconder aresta', 'crysface': 'Mostrar/esconder face do cristal', 'crysatoms': 'Mostrar/esconder \u00e1tomos',
                'hkl_000': 'hkl deve ser diferente de "0 0 0"', 'hkl_inf': 'Entre um HKL para seguir um pico', 'hkl_inf2': '(valor m\u00e1ximo plotado: {max})',
                'showHKL':'Clique para mostrar/ocultar HKL',
                'tt_s_uc': 'Clique e arraste para dar "zoom" no cristal', 
                'addAtoms': 'Adicionar um \u00e1tomo na base', 'remAtoms': 'Remover o \u00faltimo \u00e1tomo da base',
                'posx': 'Posi\u00e7\u00e3o x', 'posy': 'Posi\u00e7\u00e3o y', 'posz': 'Posi\u00e7\u00e3o z', 
                'atomtype': 'Elemento qu\u00cdmico: entre um novo para trocar',
                'save_for_latter': 'Salvar como valor padr\u00e3o para a pr\u00f3xima vez',
                'graph_settings':'Mudar algumas configura\u00e7\u00f5es do gr\u00e1fico',
                'params_settings':'Mudar algumas configura\u00e7\u00f5es destes par\u00e2metros',
                'baseatoms_settings': 'Mudar algumas configura\u00e7\u00f5es deste \u00e1tomo da base',
                'min':'m\u00ednimo',
                'max':'m\u00e1ximo',
                'param_energy_ini':'Energia inicial/padr\u00e3o',
                'param_energy_minmax':'Energia {minmax} para o slider',
                'param_CrysSize_ini':'Tamanho de cristalito inicial/padr\u00e3o',
                'param_CrysSize_minmax':'Tamanho de cristalito {minmax} para o slider',
                'lattice_param_ini':'Valor inicial/default para este par\u00e2metro',
                'lattice_param_minmax':'Valor {minmax} para o slider deste par\u00e2metro',
                'atom_size': 'Tamanho do \u00e1tomo (nu\u00famero entre 1 e 10)',
                'atom_size2': 'o tamanho do \u00e1tomo',
                'extended_cells': 'Expandir para 8 c\u00e9lulas unit\u00e1rias',
                'include_edge_atms': 'Incluir \u00e1tomos da borda da c\u00e9lula',
                'loaddata':'Abrir janela para incluir seus dados de difra\u00e7\u00e3o, no formato "2th vs. Int"',
                'loaddataproblem': 'N\u00e3o foi poss\u00cdvel carregar os dados, infelizmente. Coloque-os no formato "2th vs. Int" e tente novamente!',
                'loadfileproblem': 'n\u00e3o \u00e9 um nome de arquivo v\u00e1lido',
                'settings':'Mostrar op\u00e7\u00f5es para os dados carregados',
                'scale': 'Escala',
                'dotsize': 'Tamanho',
                'color_scatter': 'Mudar a cor destes pontos',
                'scatter_delete': 'Remover este conjunto de dados',
                'frozen_curve': 'Curva congelada',
                },
         'en':{'title':'X-ray Difraction Pattern', 'pars': u'Lattice Parameters (\u212b)',     'wvl': 'Wavelength (\u212b) and Energy (keV)',
                'size': 'Theoretical Crystallite Size (\u212b)',  'hkl': 'Miller Indexes (hkl)',    'int': 'Máximum Intensity', 
                'atm': 'Atom', 'crystal': 'Unit Cell' , 'base': 'Base Atoms', 'label_0': 'Atom of position (x,y,z) = (0,0,0)',
                'info': 'Infos', 'welcome': 'Welcome to XRD Playground', 'xpd':'Powder X-ray Diffraction Pattern', 'lattpar':'Lattice Parameters',
                'a': 'Lattice Parameter "a"',  'b': 'Lattice Parameter "b"', 'c': 'Lattice Parameter "c"', 'alpha': 'Angle between axes defined by "b" e "c"', 
                'beta': 'Angle between axes defined by "a" e "c"', 'gamma': 'Angle between axes defined by "a" e "b"',
                'complement_0': 'in \u212b', 'complement_1': 'in degrees', 'problem_a': 'is not a valid value \nfor',
                'en_0': 'energy', 'wvl_0': 'wavelength', 'size_0': 'crystal size', 'atm_0': 'atom position',
                'freeze0':'freeze curve', 'freeze1':r'{} color', 'red': 'red', 'green':'green', 'blue':'blue',
                'rescale':'Intensity rescale',
                'tt_b_pars':'Click to reset value', 'tt_s_pars':'Click and drag to change', 'tt_e_pars':'Enter value',
                'color_0':'Change this atom color',
                'crysedge': 'Show/hide edge', 'hkl_000': 'hkl should be different from "0 0 0"', 'hkl_inf': 'enter an HKL to follow a peak', 'hkl_inf2': '(maximum plotted value: {max})',
                'showHKL':'click to show/hide HKL',
                'tt_s_uc': 'Click and drag to zoom crystal', 'crysface': 'Show/hide crystal faces', 'crysatoms': 'Show/hide atoms',
                'addAtoms': 'Add one atom into the base', 'remAtoms': 'Remove last atom from base',
                'posx': 'x position', 'posy': 'y position', 'posz': 'z position', 
                'atomtype': 'Atom type: enter a new one to change',
                'save_for_latter': 'Save as new default value for next time',
                'graph_settings':'Change graph settings',
                'min':'minimum',
                'max':'maximum',
                'params_settings':"Change these parameters' settings",
                'baseatoms_settings': 'Change some settings concerning this atom',
                'param_energy_ini':'Initial/default energy',
                'param_energy_minmax':'{minmax} energy for the slider',
                'param_CrysSize_ini':'Inicial/default crystallite size',
                'param_CrysSize_minmax':'{minmax} crystallite size for the slider',
                'lattice_param_ini':'Initial/default value for this parameter',
                'lattice_param_minmax':"{minmax} value for this parameter's slider",
                'atom_size': 'Atom size (number between 1 and 10)',
                'atom_size2': 'the size of the atom',
                'extended_cells': 'Expande to 8 unitary cells',
                'include_edge_atms': 'Include atoms from edge',
                'loaddata':'Open window to include your diffraction data in "2th vs. Int" format',
                'loaddataproblem': 'data could not be loaded, sorry. Fix it using "2th vs. Int" format',
                'loadfileproblem': 'is not a valid file name',
                'settings':'Show options for loaded data',
                'scale': 'Scale',
                'dotsize': 'Size',
                'color_scatter': 'Change dot colors',
                'scatter_delete': 'Remove dataset',
                'frozen_curve': 'Frozen curve',
                },
         'es':{'title':'Patr\u00f3n de Difracci\u00f3n de Rayos X', 'pars': u'Par\u00e1metros de Red (\u212b)', 'wvl': 'Longitud de Onda (\u212b) y Energ\u00eda (keV)',
                'size': 'Tama\u00f1o Te\u00f3rico de Cristalito (\u212b)',  'hkl': '\u00cdndices de Miller (hkl)', 'int': 'Intensidad M\u00e1xima', 
                'atm': '\u00c1tomo', 'crystal': 'Celda Unitaria', 'base': '\u00c1tomos de la Base', 'label_0': '\u00c1tomo de la posici\u00f3n (x,y,z) = (0,0,0)',
                'info': 'Informaciones', 'welcome': 'Bien venido al XRD Playgroud!', 'xpd':'Patr\u00f3n de Difracci\u00f3n de Rayos X', 'lattpar': 'Par\u00e1metros de Red',
                'a': 'Par\u00e1metro "a"',  'b': 'Par\u00e1metro "b"', 'c': 'Par\u00e1metro "c"', 'alpha': '\u00c1ngulo entre ejes definidos por "b" y "c"',
                'beta': '\u00c1ngulo entre ejes definidos por "a" y "c"', 'gamma': '\u00c1ngulo entre ejes definidos por "a" y "b"', 
                'complement_0': 'en \u212b', 'complement_1': 'en grados', 'problem_a': 'no es un valor v\u00e1lido \npara',
                'en_0': 'energ\u00eda', 'wvl_0': 'longitud de onda', 'size_0': 'tama\u00f1o de cristalito', 'atm_0': 'posici\u00f3n del \u00e1tomo',
                'freeze0':'congelar curva', 'freeze1':r'color {}', 'red': 'roja', 'green':'verde', 'blue':'azul',
                'rescale':'Ajustar la escala de la intensidad',
                'tt_b_pars':'Haga clic para volver al valor inicial', 'tt_s_pars':'Haga clic y arrastre para cambiar', 'tt_e_pars':'Introduce un valor',
                'color_0':'Cambiar color del \u00e1tomo',
                'crysedge': 'Mostrar/ocultar borde', 'hkl_000': 'hkl deve ser diferente de "0 0 0"', 'hkl_inf': 'ingrese un HKL para seguir un pico', 'hkl_inf2': '(valor m\u00e1ximo trazado: {max})',
                'showHKL':'Haga clic para mostrar/ocultar HKL',
                'tt_s_uc': 'Haga clic y arrastre para acercar o alejar el cristal', 'crysface': 'Mostrar/ocultar faces', 'crysatoms': 'Mostrar/ocultar \u00e1tomos',
                'addAtoms': 'A\u00f1adir un \u00e1tomo en la base', 'remAtoms': 'Quitar el \u00faltimo \u00e1tomo de la base',
                'posx': 'posici\u00f3n x', 'posy': 'posici\u00f3n y', 'posz': 'posici\u00f3n z', 
                'atomtype': 'Tipo de \u00e1tomo: ingrese uno nuevo para cambiar',
                'save_for_latter': 'Guardar como est\00e1ndar para la pr\u00f3xima vez',
                'graph_settings':'Cambiar algunas configuraciones de la gr\u00c1fica',
                'min':'m\u00ednimo',
                'max':'m\u00e1ximo',
                'params_settings':'Cambiar algunas configuraciones de estos par\u00e1metros',
                'baseatoms_settings': 'Cambiar algunas configuraciones de este \u00e1tomo',
                'param_energy_ini':'Energ\u00eda inicial/est\00e1ndar',
                'param_energy_minmax':'Energ\u00eda {minmax} para el control deslizante',
                'param_CrysSize_ini':'Tama\u00f1o de cristalito inicial/est\00e1ndar',
                'param_CrysSize_minmax':'Tama\u00f1o de cristalito {minmax} para el control deslizante',
                'lattice_param_ini':'Valor inicial/est\00e1ndar para este par\u00e1metro',
                'lattice_param_minmax':'Valor {minmax} para el control deslizante de este par\u00e1metro',
                'atom_size': 'Tama\u00f1o del \u00e1tomo (nu\u00famero entre 1 y 10)',
                'atom_size2': 'el tama\u00f1o del \u00e1tomo',
                'extended_cells': 'Expandir para 8 celdas unitarias',
                'include_edge_atms': 'A\u00f1adir \u00e1tomos del borde de la celda unitaria',
                'loaddata':'Abrir una ventana para incluir sus datos de difracci\u00f3n en el formato "2th vs. Int"',
                'loaddataproblem': 'No fue posible cargar los datos. Col\u00f3quelos en el formato "2th vs int" e int\u00e9ntelo nuevamente',
                'loadfileproblem': 'no es un nombre de archivo v\u00e1lido',
                'settings':'Mostrar opciones para los datos cargados',
                'scale': 'Amplitud',
                'dotsize': 'Tama\u00f1o',
                'color_scatter': 'Cambiar color de los puntos',
                'scatter_delete': 'Eliminar este conjunto de datos',
                'frozen_curve': 'Curva congelada',
                }}
        self.setLanguage('en')
    def setLanguage(self, language = 'en'):
        if language not in ['br', 'es', 'en']:
            print ('language not implemented, swicthing to English')
            language = 'en'
        self.what = self.names[language]

class Colors():
    """ This is just a class that play with colors. """
    def __init__(self):
        pass
    def darken(self, color): #in format '#rrggbb'
        r = float(int(color[1:3],16))/255.
        g = float(int(color[3:5],16))/255.
        b = float(int(color[5:7],16))/255.
        h, s, v = colorsys.rgb_to_hsv(r,g,b)
        news = s/2. 
        newv = v/2.
        _newr, _newg, _newb = colorsys.hsv_to_rgb(h,news,newv)
        newr = hex(int(_newr*255))[2:4]
        newg = hex(int(_newg*255))[2:4]
        newb = hex(int(_newb*255))[2:4]
        return '#{:0>2}{:0>2}{:0>2}'.format(newr,newg,newb)
    def gray (self, color, value = 0.5):   #in format '#rrggbb' -> turning colors more gray. "value" should be between 0 (little gray) and 1 (gray)
        r = float(int(color[1:3],16))/255.
        g = float(int(color[3:5],16))/255.
        b = float(int(color[5:7],16))/255.
        h, s, v = colorsys.rgb_to_hsv(r,g,b)
        news = s - value
        if news < 0: news = 0
        newv = v
        _newr, _newg, _newb = colorsys.hsv_to_rgb(h,news,newv)
        newr = hex(int(_newr*255))[2:4]
        newg = hex(int(_newg*255))[2:4]
        newb = hex(int(_newb*255))[2:4]
        return '#{:0>2}{:0>2}{:0>2}'.format(newr,newg,newb)
    def lighten(self, color): #in format '#rrggbb'
        r = float(int(color[1:3],16))/255.
        g = float(int(color[3:5],16))/255.
        b = float(int(color[5:7],16))/255.
        h, s, v = colorsys.rgb_to_hsv(r,g,b)
        news = s/2. 
        newv = 0.5+v/2.
        _newr, _newg, _newb = colorsys.hsv_to_rgb(h,news,newv)
        newr = hex(int(_newr*255))[2:4]
        newg = hex(int(_newg*255))[2:4]
        newb = hex(int(_newb*255))[2:4]
        return '#{:0>2}{:0>2}{:0>2}'.format(newr,newg,newb)

class Defaults():
    """ This is a class that creates a default initialization file, when it does not exist, or update it when the user asks to. """
    def __init__(self):
        # 2th range
        self.tth_min = 5
        self.tth_max = 65
        self.tth_step = 0.04
        # hkl max
        self.h_max = 4
        self.k_max = 4
        self.l_max = 4
        # initial energy and energy range
        self.E_ini = 8
        self.E_min = 4
        self.E_max = 20
        # initial crystal size and crystal size range
        self.CrysSize_ini = 500
        self.CrysSize_min = 15
        self.CrysSize_max = 1000
        # initial lattice parameters
        self.a_ini = 5.640
        self.b_ini = 5.640
        self.c_ini = 5.640
        self.alpha_ini = 90
        self.beta_ini = 90
        self.gamma_ini = 90
        # initial lattice parameters minumum
        self.a_min = 2
        self.b_min = 2
        self.c_min = 2
        self.alpha_min = 40
        self.beta_min = 40
        self.gamma_min = 40
        # initial lattice parameters maximum
        self.a_max = 13
        self.b_max = 13
        self.c_max = 13
        self.alpha_max = 150
        self.beta_max = 150
        self.gamma_max = 150
        # initial base atom
        self.atom_ini = "Fe"
        self.atom_ini_size = 300
        # additional atoms
        self.additional_atoms = []
        self.additional_atoms_positions = []
        self.additional_atoms_sizes = []
        # base atoms
        self.baseAtoms = []
        self.baseAtoms_positions = []
        self.baseAtoms_sizes = []
        # language (one of 'en', 'br', 'es')
        self.language = 'en'
        # micelaneous
        self.font = "Arial"
        self.fontsize = 12
        ####
        self.default = {'tth_min': self.tth_min, 
                         'tth_max': self.tth_max,
                         'tth_step':self.tth_step,
                         'h_max':   self.h_max,
                         'k_max':   self.k_max,
                         'l_max':   self.l_max,
                         'E_ini':   self.E_ini,
                         'E_min':   self.E_min,
                         'E_max':   self.E_max,
                         'CrysSize_ini': self.CrysSize_ini,
                         'CrysSize_min': self.CrysSize_min,
                         'CrysSize_max': self.CrysSize_max,
                         'a_ini':   self.a_ini,
                         'b_ini':   self.b_ini,
                         'c_ini':   self.c_ini,
                         'alpha_ini':   self.alpha_ini,
                         'beta_ini':    self.beta_ini,
                         'gamma_ini':   self.gamma_ini,
                         'a_min':   self.a_min,
                         'b_min':   self.b_min,
                         'c_min':   self.c_min,
                         'alpha_min':   self.alpha_min,
                         'beta_min':    self.beta_min,
                         'gamma_min':   self.gamma_min,
                         'a_max':   self.a_max,
                         'b_max':   self.b_max,
                         'c_max':   self.c_max,
                         'alpha_max':   self.alpha_max,
                         'beta_max':    self.beta_max,
                         'gamma_max':   self.gamma_max,
                         'atom_ini':    self.atom_ini,
                         'atom_ini_size':    self.atom_ini_size,
                         'additional_atoms': self.additional_atoms,
                         'additional_atoms_positions':  self.additional_atoms_positions,
                         'additional_atoms_sizes':      self.additional_atoms_sizes,
                         'baseAtoms':                   self.baseAtoms,
                         'baseAtoms_positions':         self.baseAtoms_positions,
                         'baseAtoms_sizes':             self.baseAtoms_sizes,
                         'language':    self.language,
                         'font':        self.font,
                         'fontsize':    self.fontsize}
    def createDefault(self):
        lstout = ["# default values for pxrd"]
        # 2th range
        list = ["# 2th range", "tth_min = {}".format(self.default['tth_min']), "tth_max = {}".format(self.default['tth_max']), "tth_step = {}".format(self.default['tth_step'])]
        for i in list: lstout.append(i)
        # hkl max
        list = ["# hkl max","h_max = {}".format(self.default['h_max']),"k_max = {}".format(self.default['k_max']),"l_max = {}".format(self.default['l_max'])]
        for i in list: lstout.append(i)
        # initial energy and energy range
        list = ["# energy", "E_ini = {}".format(self.default['E_ini']), "E_min = {}".format(self.default['E_min']), "E_max = {}".format(self.default['E_max'])]
        for i in list: lstout.append(i)
        # initial crystal size and crystal size range
        list = ["# crystal size", "CrysSize_ini = {}".format(self.default['CrysSize_ini']), "CrysSize_min = {}".format(self.default['CrysSize_min']), "CrysSize_max = {}".format(self.default['CrysSize_max'])]
        for i in list: lstout.append(i)
        # initial lattice parameters
        list = ["# initial lattice parameters", "a_ini = {}".format(self.default['a_ini']), "b_ini = {}".format(self.default['b_ini']), "c_ini = {}".format(self.default['c_ini']), "alpha_ini = {}".format(self.default['alpha_ini']), "beta_ini = {}".format(self.default['beta_ini']), "gamma_ini = {}".format(self.default['gamma_ini'])]
        for i in list: lstout.append(i)
        # initial lattice parameters minumum
        list = ["# initial lattice parameters minimum", "a_min = {}".format(self.default['a_min']), "b_min = {}".format(self.default['b_min']), "c_min = {}".format(self.default['c_min']), "alpha_min = {}".format(self.default['alpha_min']), "beta_min = {}".format(self.default['beta_min']), "gamma_min = {}".format(self.default['gamma_min'])]
        for i in list: lstout.append(i)
        # initial lattice parameters maximum
        list = ["# initial lattice parameters maximum", "a_max = {}".format(self.default['a_max']), "b_max = {}".format(self.default['b_max']), "c_max = {}".format(self.default['c_max']), "alpha_max = {}".format(self.default['alpha_max']), "beta_max = {}".format(self.default['beta_max']), "gamma_max = {}".format(self.default['gamma_max'])]
        for i in list: lstout.append(i)
        # initial base atom
        list = ["# initial base atom", "atom_ini = {}".format(self.default['atom_ini']), "atom_ini_size = {}".format(self.default['atom_ini_size'])]
        for i in list: lstout.append(i)
        # additional atoms
        list = ["# additional atoms", "additional_atoms = {}".format(self.default['additional_atoms']), "additional_atoms_positions = {}".format(self.default['additional_atoms_positions']),  "additional_atoms_sizes = {}".format(self.default['additional_atoms_sizes'])]
        for i in list: lstout.append(i)
        # base atoms
        list = ["# base atoms", "baseAtoms = {}".format(self.default['baseAtoms']), "baseAtoms_positions = {}".format(self.default['baseAtoms_positions']),  "baseAtoms_sizes = {}".format(self.default['baseAtoms_sizes'])]
        for i in list: lstout.append(i)
         # language
        list = ["# initial language", "language = {}".format(self.default['language'])]
        for i in list: lstout.append(i)
        # micelaneous
        list = ["# other options", "font_ini = {}".format(self.default['font']), "fontsize_ini = {}".format(self.default['fontsize'])]
        for i in list: lstout.append(i)
        with open("pxrd.defaults", "w") as f:
            f.write("\n".join(lstout))
            f.close()
    def loadDefault(self):
        self.inis = {}
        # all these beacuse I dind not want to use "exec" 
        with open("pxrd.defaults", "r") as f:
            for line in f:
                if line[0] != "#":
                    (key, val) = line.split("=")
                    key = key.replace(" ", "")
                    val = val.replace(" ", "").replace("\n", "")
                    if key in ["additional_atoms", "additional_atoms_sizes", "baseAtoms", "baseAtoms_sizes"]:
                        if val[1:-1] == '': val = []
                        else:
                            val = val[1:-1].replace("'", "").replace('"','').split(",")
                            val0 = []
                            try:
                                for i in val: val0.append(float(i))
                                val = val0
                            except:
                                pass
                    elif key in ["additional_atoms_positions", "baseAtoms_positions"]:
                        if val[1:-1] == '': val = []
                        else: 
                            val0 = []
                            val1 = val[2:-2].split("],[")
                            for i in val1:
                                val2 = i.split(",")
                                val3 = []
                                try: 
                                    for i in val2: val3.append(float(i))
                                    val2 = val3
                                except:
                                    pass
                                val0.append(val2)
                            val = val0
                    self.inis[key] = val
            f.close()
        #for i in (self.inis): print ('{} : {}'.format(i, self.inis[i]))
        return self.inis
    def delDefault(self):
        if os.path.isfile("pxrd.defaults"): os.remove("pxrd.defaults")
        
class DoubleSlider(QSlider):
    """ This double slider class is used to make finer the step of the sliders. """
    doubleValueChanged = pyqtSignal(float)

    def __init__(self, decimals=3, *args, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._multi = 10 ** decimals

        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(int(value * self._multi))

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(int(value * self._multi))

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(int(value * self._multi))

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value):
        super(DoubleSlider, self).setValue(int(value * self._multi))

class Icons():
    """ This class I use only to draw some figures for the program, since I did not want to take ready ones """
    def __init__(self):
        import matplotlib

        fig, ax = plt.subplots()
        n = 30.
        w = 5
        x = np.arange(n)
        y0 = np.exp((x-n/2.-n/15.)*(n/2.+n/15.-x)/(2.*n*n/100.))
        y1 = np.exp((x-n/2.+n/15.)*(n/2.-n/15.-x)/(2.*n*n/100.))
        
        line0, = ax.plot(x,y0, c= 'k', lw = w)
        line1, = ax.plot(x,y0, c= 'k', lw = w)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_ylim(-0.2*y0.max()+y0.min(),y0.max()*1.4)
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(w*2/3.)
        self.name = []
        for j,i in enumerate(['red', 'green', 'blue']):
            line1.set_ydata(y1)
            line1.set_color(i)
            fig.set_size_inches(1,1)
            self.name.append(matplotlib.get_data_path() + r"\images\freeze_" + '{}'.format(i))
            plt.savefig(self.name[j] + '_large.png', dpi=48)
            plt.savefig(self.name[j] + '.png', dpi=24)
        y2 = 0.5*np.exp((x-n/2.)*(n/2.-x)/(2.*n*n/100.))
        y3 = 0.3+0.8*np.exp((x-n/2.)*(n/2.-x)/(2.*n*n/100.))
        line0.set_ydata(y3)
        line1.set_ydata(y2)
        line1.set_color('#aaaaaa')
        name = matplotlib.get_data_path() + r"\images\rescale"
        self.name.append(name)
        plt.savefig(name + '_large.png', dpi=48)
        plt.savefig(name + '.png', dpi=24)
        nn = 30
        x_new = np.arange(nn)
        y_new = np.random.random(len(x_new))*0.05 + np.exp((x_new-nn/4)*(nn/4.-x_new)/(2.*2)) + 0.5*np.exp((x_new-3.2*nn/4)*(3.2*nn/4.-x_new)/(2.*2))
        line0.set_data([],[])
        line1.set_data([],[])
        scat = ax.scatter (x_new, y_new, c = 'k', s = 0.2)
        plt.errorbar(x_new, y_new, yerr=0.11, fmt=".", color = 'k')
        name = matplotlib.get_data_path() + r"\images\loaddata"
        self.name.append(name)
        plt.savefig(name + '_large.png', dpi=48)
        plt.savefig(name + '.png', dpi=24)
        
        scat.remove()
        fig.clf()
        theta = np.arange(0,2*np.pi+0.02,0.02)
        def f(t): return 0.15*np.cos(8*t) + 1
        
        def f2(t): return 0.8
        
        fig, ax = plt.subplots()
        for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(w*2/3.)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        
        
        axes = fig.add_axes([0.2, 0.2, 0.6, 0.6], projection ='polar')
        axes.xaxis.set_ticks([])
        axes.yaxis.set_ticks([])
        
        axes.set_rmax(1.7)
        axes.plot (theta, f(theta), c = 'k')
        plt.fill_between(x = theta, y1 = f(theta), color = 'k')
        plt.fill_between(x = theta, y1 = 0.5, color = '#ffffff')

        
        for axis in ['polar']: axes.spines[axis].set_linewidth(0*2/3.)
        
        fig.set_size_inches(1,1)
        name = matplotlib.get_data_path() + r"\images\settings"
        self.name.append(name)
        plt.savefig(name + '_large.png', dpi=48)
        plt.savefig(name + '.png', dpi=24)
        fig.clf()

class Freeze(ToolToggleBase):
    """Freeze simulation, creating a new colored curve above the main curve. Used to compare simulations."""
    default_toggled = False
    def __init__(self, *args, curve, plot, main_plot, description, **kwargs):
        self.curve_dict = {0:'r', 1:'g', 2:'b'}
        self.curve_color = {0:'red', 1:'green', 2:'blue'}
        self.plot = plot
        self.main_plot = main_plot
        super().__init__(*args, **kwargs)
        self.default_keymap = self.curve_dict[curve]
        self.description = description['freeze0'] + ' (' + description['freeze1'].format(description[self.curve_color[curve]]) + ')'
        self.image = a.name[curve]
    def enable(self, *args):
        self.set_freeze(True)
    def disable(self, *args):
        self.set_freeze(False)
    def set_freeze(self, state):
        if state:
            self.plot.set_data(self.main_plot.get_xdata(),self.plot.scale*self.main_plot.get_ydata())
        self.plot.set_visible(state)
        self.figure.canvas.draw()

class Rescale_y(ToolBase):
    """Rescale the pXRD figure, showing all data."""
    default_keymap = 'y'
    
    def __init__(self, *args, func, main_plot, description, **kwargs):
        super().__init__(*args, **kwargs)
        self.image = a.name[3]
        self.func = func
        self.main_plot = main_plot
        self.lang = Language()
        self.description = description
        #print (self.image)
    def trigger(self, *args, **kwargs):
        self.func()
        self.figure.canvas.draw()

class Load_Data(ToolBase):
    """Open a dialog which permit the user to load a file in the format two-theta vs. intensity. """
    default_keymap = 'q'
    
    def __init__(self, *args, xrdp, description, **kwargs):
        super().__init__(*args, **kwargs)
        self.xrdp = xrdp
        
       
        self.image = a.name[4]
        self.description = description
    def trigger(self, *args, **kwargs):
        dialog = QFileDialog(self.xrdp)
        fname = inspect.getframeinfo(inspect.currentframe()).filename
        path = os.path.dirname(os.path.abspath(fname))        
        dialog.setDirectory(path)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setViewMode(QFileDialog.ViewMode.List)
        filename = dialog.getOpenFileName()[0]
        
        if not os.path.isfile(filename): 
            print (f'{filename}' + self.xrdp.l.what['loadfileproblem'])
            return
        else:
            loaded = False
            for i in range(10):
                try:
                    x, y = np.transpose(np.loadtxt(filename, skiprows = i, usecols = (0,1)))
                    loaded = True
                except:
                    pass
            if loaded:
                label = os.path.basename(filename)
                number = 1
                while (label in self.xrdp.userdata_dict.keys()):
                    test = label + f'#{number}'
                    if test in self.xrdp.userdata_dict.keys():
                        number +=1
                    else:
                        label = test
                self.xrdp.userdata_dict.update({label:{}})
                self.xrdp.userdata_dict[label].update({'datax':np.array(x), 'datay':np.array(y)})
                self.xrdp.userdata_dict[label].update({'scatters':self.xrdp.pXRDax.scatter(x,y, label = os.path.basename(filename))})
                self.xrdp.userdata_dict[label].update({'scale': 1, 'size': 36})
                
                #self.xrdp.userdata.append([np.array(x),np.array(y)])
                #self.xrdp.userdata_scatters.append(self.xrdp.pXRDax.scatter(x,y, label = os.path.basename(filename)))
                #self.xrdp.userdata_scales.append(1)
                #self.xrdp.userdata_sizes.append([])
                
                self.xrdp.XPDcanvas.draw_idle()
            else:
                print (f'{self.xrdp.l.what["loaddataproblem"]}')

class Settings_Data(ToolBase):
    """ Button action which opens the window below  """
    default_keymap = 'o'
    
    def __init__(self, *args, xrdp, description, func, **kwargs):
        super().__init__(*args, **kwargs)
        self.xrdp = xrdp
        
        self.image = a.name[5]
        self.description = description
        self.func = func
        
    def trigger(self, *args, **kwargs):
        noc, dict, func_ = self.func()
        if True:
            self.w1 = PopUpUserDataOpt(self, self.xrdp, noc, 'Settings User Data', dict, func_)
            self.w1.setGeometry(QRect(100, 100, 315, 5 + noc*30))
            self.w1.show()
        
class PopUpUserDataOpt(QWidget):
    """ Opens a window which permits to change some settings concening the user data and freezed simulations. """
    
    def __init__(self, setdata, xrdp, noc, title, dict, func):
        self.colors = Colors()
        self.xrdp = xrdp
        
        self.labels = []
        self.old_colors = []
        for i in list(dict.keys()):
            self.labels.append(dict[i]['label'])
            self.old_colors.append(dict[i]['color'])
        QWidget.__init__(self)
        self.layout = QVBoxLayout(self)
        self.setWindowTitle(title)
        self.file_dict = {}
        self.frozen_dict = {}
        
        self.layout_insideGroupBox = {}
        
        for j, i in enumerate(['red', 'green' , 'blue']):
            
            self.frozen_dict.update({i:{}})
            
            string = xrdp.l.what["freeze1"].format(i)
            self.frozen_dict[i].update({'groupbox':QGroupBox(f'{xrdp.l.what["frozen_curve"]} ({string})')})
            self.frozen_dict[i]['groupbox'].setContentsMargins(0,5,0,0)
            self.frozen_dict[i].update({'layout':QGridLayout(self.frozen_dict[i]['groupbox'])})
            self.frozen_dict[i]['layout'].setAlignment(Qt.AlignLeft)

            self.frozen_dict[i].update({'label_scale':QLabel()})
            self.frozen_dict[i]['label_scale'].setFont(QFont(xrdp.font,xrdp.fontsize-2))
            self.frozen_dict[i]['label_scale'].setText(f'{xrdp.l.what["scale"]} ')
            label_opts = self.Label_StyleSheet(i)
            self.frozen_dict[i]['label_scale'].setStyleSheet(label_opts)
            self.frozen_dict[i]['label_scale'].setFixedWidth(65)
            self.frozen_dict[i]['label_scale'].setFixedHeight(18)
            self.frozen_dict[i]['label_scale'].setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            self.frozen_dict[i]['label_scale'].setContentsMargins(0,0,0,0)

            self.frozen_dict[i].update({'lineedit_scale':QLineEdit(f'{str(xrdp.colored_plots[j].scale)}')})
            entry_opts = self.LineEdit_StyleSheet(self.colors.lighten(rgb2hex(xrdp.colored_plots[j].get_color())))
            self.frozen_dict[i]['lineedit_scale'].setStyleSheet(entry_opts)
            self.frozen_dict[i]['lineedit_scale'].setFont(QFont(xrdp.font,xrdp.fontsize-3))
            self.frozen_dict[i]['lineedit_scale'].setMaxLength(12)
            self.frozen_dict[i]['lineedit_scale'].editingFinished.connect(lambda numb = j, label = i: self.change_scale(numb, label))
            self.frozen_dict[i]['lineedit_scale'].setFixedWidth(40)
            self.frozen_dict[i]['lineedit_scale'].setFixedHeight(18)
            self.frozen_dict[i]['lineedit_scale'].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.frozen_dict[i]['lineedit_scale'].setContentsMargins(0,0,0,0)

            self.layout.addWidget(self.frozen_dict[i]['groupbox'])  
            self.frozen_dict[i]['layout'].addWidget(self.frozen_dict[i]['label_scale'],1,1,1,1)
            self.frozen_dict[i]['layout'].addWidget(self.frozen_dict[i]['lineedit_scale'],1,2,1,1)

            self.frozen_dict[i].update({'label_dotsize':QLabel()})
            self.frozen_dict[i]['label_dotsize'].setFont(QFont(xrdp.font,xrdp.fontsize-2))
            self.frozen_dict[i]['label_dotsize'].setText('{}'.format(xrdp.l.what['dotsize']))
            label_opts = self.Label_StyleSheet(i)
            self.frozen_dict[i]['label_dotsize'].setStyleSheet(label_opts)
            self.frozen_dict[i]['label_dotsize'].setFixedWidth(65)
            self.frozen_dict[i]['label_dotsize'].setFixedHeight(18)
            self.frozen_dict[i]['label_dotsize'].setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            self.frozen_dict[i]['label_dotsize'].setContentsMargins(0,0,0,0)
            
            size = xrdp.colored_plots[j].get_markersize()
            self.frozen_dict[i].update({'lineedit_dotsize':QLineEdit(f'{size}')})
            entry_opts = self.LineEdit_StyleSheet(self.colors.lighten(rgb2hex(xrdp.colored_plots[j].get_color())))
            self.frozen_dict[i]['lineedit_dotsize'].setStyleSheet(entry_opts)
            self.frozen_dict[i]['lineedit_dotsize'].setFont(QFont(xrdp.font,xrdp.fontsize-3))
            self.frozen_dict[i]['lineedit_dotsize'].setMaxLength(12)
            self.frozen_dict[i]['lineedit_dotsize'].editingFinished.connect(lambda numb = j, label = i: self.change_size(numb, label))
            self.frozen_dict[i]['lineedit_dotsize'].setFixedWidth(40)
            self.frozen_dict[i]['lineedit_dotsize'].setFixedHeight(18)
            self.frozen_dict[i]['lineedit_dotsize'].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.frozen_dict[i]['lineedit_dotsize'].setContentsMargins(0,0,0,0)
            
            self.frozen_dict[i]['layout'].addWidget(self.frozen_dict[i]['label_dotsize'],1,3,1,1)
            self.frozen_dict[i]['layout'].addWidget(self.frozen_dict[i]['lineedit_dotsize'],1,4,1,1)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        for i in range(noc):
            file = self.labels[i]
            self.file_dict.update({file:{}})
            
            self.file_dict[file].update({'groupbox':QGroupBox(f'{file}')})
            self.file_dict[file]['groupbox'].setContentsMargins(0,5,0,0)
            self.file_dict[file].update({'layout':QGridLayout(self.file_dict[file]['groupbox'])})
            self.file_dict[file]['layout'].setAlignment(Qt.AlignLeft)
            
            self.file_dict[file].update({'label_scale':QLabel()})
            self.file_dict[file]['label_scale'].setFont(QFont(xrdp.font,xrdp.fontsize-2))
            self.file_dict[file]['label_scale'].setText('{}'.format(xrdp.l.what['scale']))
            label_opts = self.Label_StyleSheet(self.old_colors[i])
            self.file_dict[file]['label_scale'].setStyleSheet(label_opts)
            self.file_dict[file]['label_scale'].setFixedWidth(65)
            self.file_dict[file]['label_scale'].setFixedHeight(18)
            self.file_dict[file]['label_scale'].setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            self.file_dict[file]['label_scale'].setContentsMargins(0,0,0,0)
            
            self.file_dict[file].update({'lineedit_scale':QLineEdit(f'{dict[file]["scale"]}')})
            entry_opts = self.LineEdit_StyleSheet(self.colors.lighten(self.old_colors[i]))
            self.file_dict[file]['lineedit_scale'].setStyleSheet(entry_opts)
            self.file_dict[file]['lineedit_scale'].setFont(QFont(xrdp.font,xrdp.fontsize-3))
            self.file_dict[file]['lineedit_scale'].setMaxLength(12)
            self.file_dict[file]['lineedit_scale'].editingFinished.connect(lambda func = func, numb = i, label = file: self.change(numb, func, label))
            self.file_dict[file]['lineedit_scale'].setFixedWidth(40)
            self.file_dict[file]['lineedit_scale'].setFixedHeight(18)
            self.file_dict[file]['lineedit_scale'].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            #self.file_dict[file]['lineedit_scale'].setToolTip(tooltips[i])
            self.file_dict[file]['lineedit_scale'].setContentsMargins(0,0,0,0)
            
            self.file_dict[file].update({'label_dotsize':QLabel()})
            self.file_dict[file]['label_dotsize'].setFont(QFont(xrdp.font,xrdp.fontsize-2))
            self.file_dict[file]['label_dotsize'].setText('{}'.format(xrdp.l.what['dotsize']))
            label_opts = self.Label_StyleSheet(self.old_colors[i])
            self.file_dict[file]['label_dotsize'].setStyleSheet(label_opts)
            self.file_dict[file]['label_dotsize'].setFixedWidth(65)
            self.file_dict[file]['label_dotsize'].setFixedHeight(18)
            self.file_dict[file]['label_dotsize'].setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            self.file_dict[file]['label_dotsize'].setContentsMargins(0,0,0,0)
            
            self.file_dict[file].update({'lineedit_dotsize':QLineEdit(f'{dict[file]["size"]}')})
            entry_opts = self.LineEdit_StyleSheet(self.colors.lighten(self.old_colors[i]))
            self.file_dict[file]['lineedit_dotsize'].setStyleSheet(entry_opts)
            self.file_dict[file]['lineedit_dotsize'].setFont(QFont(xrdp.font,xrdp.fontsize-3))
            self.file_dict[file]['lineedit_dotsize'].setMaxLength(12)
            self.file_dict[file]['lineedit_dotsize'].editingFinished.connect(lambda func = func, numb = i, label = file: self.change2(numb, func, label))
            self.file_dict[file]['lineedit_dotsize'].setFixedWidth(40)
            self.file_dict[file]['lineedit_dotsize'].setFixedHeight(18)
            self.file_dict[file]['lineedit_dotsize'].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            #self.file_dict[file]['lineedit_scale'].setToolTip(tooltips[i])
            self.file_dict[file]['lineedit_dotsize'].setContentsMargins(0,0,0,0)
            
            self.file_dict[file].update({'scatter_color':QPushButton('')})
            self.file_dict[file]['scatter_color'].clicked.connect(lambda tf, label = file, numb = i: self.set_color(tf, label, numb))
            button_opts = self.PushButton_StyleSheet(self.old_colors[i])
            self.file_dict[file]['scatter_color'].setStyleSheet(button_opts)
            self.file_dict[file]['scatter_color'].setToolTip(xrdp.l.what['color_scatter'])
            icon = qta.icon('msc.symbol-color', color='k', scale_factor = 1.1)
            self.file_dict[file]['scatter_color'].setFixedWidth(17)
            self.file_dict[file]['scatter_color'].setIcon(icon)
            
            self.file_dict[file].update({'scatter_delete':QPushButton('')})
            self.file_dict[file]['scatter_delete'].clicked.connect(lambda tf, label = file, numb = i: self.remove_data(tf, label, numb))
            button_opts = self.PushButton_StyleSheet(self.old_colors[i])
            self.file_dict[file]['scatter_delete'].setStyleSheet(button_opts)
            self.file_dict[file]['scatter_delete'].setToolTip(xrdp.l.what['scatter_delete'])
            icon = qta.icon('msc.trash', color='k', scale_factor = 1.1)
            self.file_dict[file]['scatter_delete'].setFixedWidth(17)
            self.file_dict[file]['scatter_delete'].setIcon(icon)

            
            self.layout.addWidget(self.file_dict[file]['groupbox'])  
            self.file_dict[file]['layout'].addWidget(self.file_dict[file]['label_scale'],1,1,1,1)
            self.file_dict[file]['layout'].addWidget(self.file_dict[file]['lineedit_scale'],1,2,1,1)
            self.file_dict[file]['layout'].addWidget(self.file_dict[file]['label_dotsize'],1,3,1,1)
            self.file_dict[file]['layout'].addWidget(self.file_dict[file]['lineedit_dotsize'],1,4,1,1)
            self.file_dict[file]['layout'].addWidget(self.file_dict[file]['scatter_color'],1,5,1,1)
            self.file_dict[file]['layout'].addWidget(self.file_dict[file]['scatter_delete'],1,6,1,1)
        
            
  
        self.xrdp = xrdp
    def Label_StyleSheet(self, color):
        a = 'QLabel    {border: 0px solid black; border-radius: 2px; padding: 0px; margin: 0px;'
        b = 'background-color: {}'.format(color) + '; min-width: 15ex; min-height: 17 px;}'#;, min-height: 8ex;}'
        return a+b
    def LineEdit_StyleSheet(self,color):
        a = 'QLineEdit {border: 1px solid #666666; border-radius: 2px; padding: 0px; background: '+'{};'.format(color)
        b = 'selection-background-color: {};'.format(color)+' margin: 0px; width: 40px; min-height: 17 px;}'
        return a+b
    def PushButton_StyleSheet(self, color):
        a = 'QPushButton {border: 1px solid black; background-color: '+ f'{color};' 
        b = 'border-radius: 2px; padding: 0px; min-width: 25px; min-height: 17 px;}'
        return a+b
    def change (self, numb, func, label):
        #old_value = self.xrdp.userdata_scales[numb]
        old_value = self.xrdp.userdata_dict[label]['scale']
        entry_value = self.file_dict[self.labels[numb]]['lineedit_scale'].text()
        try:
            value = float(entry_value)
        except:
            print (f'{value} is not a valid value for the scale')
            self.file_dict[self.labels[numb]]['lineedit_scale'].setText(f'{str(old_value)}')
            return 
        #self.xrdp.userdata_scales[numb] = value
        self.xrdp.userdata_dict[label].update({'scale': value})
        func(numb, label)
    def change2 (self, numb, func, label):
        #old_value = self.xrdp.userdata_sizes[numb]
        old_value = self.xrdp.userdata_dict[label]['size']
        entry_value = self.file_dict[self.labels[numb]]['lineedit_dotsize'].text()
        try:
            value = float(entry_value)
        except:
            print (f'{value} is not a valid value for the dot size')
            self.file_dict[self.labels[numb]]['lineedit_dotsize'].setText(f'{str(old_value)}')
            return 
        #self.xrdp.userdata_sizes[numb] = value
        self.xrdp.userdata_dict[label].update({'size': value})
        func(numb, label)
    def change_scale(self, who, label):
        old_value = self.xrdp.colored_plots[who].scale
        entry_value = self.frozen_dict[label]['lineedit_scale'].text()
        try:
            value = float(entry_value)
        except:
            print (f'{value} is not a valid value for the the scale')
            self.frozen_dict[label]['lineedit_scale'].setText(f'{str(old_value)}')
            return 
        self.xrdp.colored_plots[who].scale = value
        self.xrdp.colored_plots[who].set_ydata(self.xrdp.colored_plots[who].get_ydata()*value/old_value)
        self.xrdp.XPDcanvas.draw_idle()
        
    def change_size(self, who, label):
        old_value = self.xrdp.colored_plots[who].get_markersize()
        entry_value = self.frozen_dict[label]['lineedit_dotsize'].text()
        try:
            value = float(entry_value)
        except:
            print (f'{value} is not a valid value for the the dotsize')
            self.frozen_dict[label]['lineedit_dotsize'].setText(f'{str(old_value)}')
            return 
        #sizes = np.ones(len(self.xrdp.colored_plots[who].get_xdata()))*value
        self.xrdp.colored_plots[who].set_markersize(value)
        #self.xrdp.colored_plots[who].set_ydata(self.xrdp.colored_plots[who].get_ydata()*value/old_value)
        self.xrdp.XPDcanvas.draw_idle()
        
        
    def set_color(self, tf, file, numb):
        old_color = self.old_colors[numb]
        color = QColorDialog.getColor().name()
        if color == '#000000': return
        self.file_dict[file]['label_scale'].setStyleSheet(self.Label_StyleSheet(color))
        self.file_dict[file]['lineedit_scale'].setStyleSheet(self.LineEdit_StyleSheet(self.colors.lighten(color)))
        self.file_dict[file]['label_dotsize'].setStyleSheet(self.Label_StyleSheet(color))
        self.file_dict[file]['lineedit_dotsize'].setStyleSheet(self.LineEdit_StyleSheet(self.colors.lighten(color)))
        self.file_dict[file]['scatter_color'].setStyleSheet(self.PushButton_StyleSheet(color))
        self.file_dict[file]['scatter_delete'].setStyleSheet(self.PushButton_StyleSheet(color))
        #self.xrdp.userdata_scatters[numb].set_facecolors(color)
        self.xrdp.userdata_dict[file]['scatters'].set_facecolors(color)
        self.xrdp.XPDcanvas.draw_idle()





    def remove_data(self, tf, file, numb):

        self.xrdp.userdata_dict[file]['scatters'].remove()
        self.xrdp.userdata_dict.pop(file)
        
        self.file_dict[file]['layout'].removeWidget(self.file_dict[file]['label_scale'])
        self.file_dict[file]['label_scale'].setParent(None)
        self.file_dict[file]['layout'].removeWidget(self.file_dict[file]['lineedit_scale'])
        self.file_dict[file]['lineedit_scale'].setParent(None)
        self.file_dict[file]['layout'].removeWidget(self.file_dict[file]['label_dotsize'])
        self.file_dict[file]['label_dotsize'].setParent(None)
        self.file_dict[file]['layout'].removeWidget(self.file_dict[file]['lineedit_dotsize'])
        self.file_dict[file]['lineedit_dotsize'].setParent(None)
        self.file_dict[file]['layout'].removeWidget(self.file_dict[file]['scatter_color'])
        self.file_dict[file]['scatter_color'].setParent(None)
        
        self.layout.removeWidget(self.file_dict[file]['groupbox'])
        self.file_dict[file]['groupbox'].setParent(None)
        
        self.file_dict.pop(file)
        self.xrdp.XPDcanvas.draw_idle()
    
class PopUpOpt(QWidget):
    """Window which permits to modify a bunch of thing, depending on who called it. """
    def __init__(self, xrdp, options, functions, names, tooltips, types, min, max, color, title, save):
        QWidget.__init__(self)
        self.layout = QVBoxLayout(self)
        self.setWindowTitle(title)
        self.GroupBox = QGroupBox('')
        self.GroupBox.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.GroupBox)        
        
        self.colors = Colors()
        
        self.Groupbox_layout = QGridLayout(self.GroupBox)
        self.Groupbox_layout.setAlignment(Qt.AlignLeft)
        
        self.keys = list(options.keys())
        self.options = options
        self.min = min
        self.max = max
        #self.related_funcs = {'linestyle':spgf.linestyle_m, 'linewidth':spgf.linewidth_m, 'label':spgf.label_m, 'multiplier':spgf.multiplier_m}
        
        self.Labels = {}
        self.Entries = {}
        self.Buttons = {}
        for j, i in enumerate(self.keys):
            self.Labels.update({i:QLabel()})
            self.Labels[i].setFont(QFont(xrdp.font,xrdp.fontsize-2))
            self.Labels[i].setText('{}'.format(names[i]))
            label_opts = self.Label_StyleSheet(color[i])
            self.Labels[i].setStyleSheet(label_opts)
            self.Labels[i].setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            self.Labels[i].setContentsMargins(0,0,0,0)

            self.Entries.update({i:QLineEdit('{}'.format(self.options[i]))})
            entry_opts = self.LineEdit_StyleSheet(self.colors.lighten(color[i]))
            self.Entries[i].setStyleSheet(entry_opts)
            self.Entries[i].setFont(QFont(xrdp.font,xrdp.fontsize-3))
            self.Entries[i].setMaxLength(12)
            self.Entries[i].editingFinished.connect(lambda who = i: self.change(who, functions[who], types[who]))
            self.Entries[i].setFixedWidth(60)
            self.Entries[i].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.Entries[i].setToolTip(tooltips[i])
            self.Entries[i].setContentsMargins(0,0,0,0)
            
            if save[i]:
                self.Buttons.update({i:QPushButton('')})
                self.Buttons[i].clicked.connect(lambda checked, who = i: self.save_as_default(who))
                button_opts = self.PushButton_StyleSheet(color[i])
                self.Buttons[i].setStyleSheet(button_opts)
                self.Buttons[i].setToolTip(xrdp.l.what['save_for_latter'])
                icon = qta.icon('ri.save-3-line', color='k', scale_factor = 1.1)
                self.Buttons[i].setIcon(icon)

            self.Groupbox_layout.addWidget(self.Labels[i],j,0,1,1)
            self.Groupbox_layout.addWidget(self.Entries[i],j,1,1,1)
            if save[i]:
                self.Groupbox_layout.addWidget(self.Buttons[i],j,2,1,1)

        self.xrdp = xrdp
    def Label_StyleSheet(self, color):
        a = 'QLabel {border: 1px solid ' + '{}'.format('#666666') + '; border-radius: 2px; padding: 0px;'
        b = 'background-color: {}'.format(color) + '; min-width: 27px;}'
        return a+b
    def LineEdit_StyleSheet(self,color):
        a = 'QLineEdit {border: 1px solid #666666; border-radius: 2px; padding: 0px; background: '+'{};'.format(color)
        b = 'selection-background-color: {};'.format(color)+' padding: -0px -0px -0px -0px;margin: -0px -0px -0px -0px;width: 40px;}'
        return a+b
    def PushButton_StyleSheet(self, color):
        a = 'QPushButton {background-color: '+'{}; border: 1px solid;'.format(color) 
        b = 'border-radius: 2px; border-color: #666666; max-width: 18px; padding: 0;}'
        return a+b
    def change (self, who, func, type):

        old_value = self.options[who]
        entry_value = self.Entries[who].text()
        if type in ['float', 'int']:
            try:
                value = float(entry_value)
                if type == 'int':
                    value = int(value)
                if value < self.min[who] or value > self.max[who]:
                    print ('{} is out of bonds'.format(entry_value))
                    self.Entries[who].setText('{}'.format(old_value))
                    return
            except:
                print ('{} is not a valid value'.format(entry_value))
                self.Entries[who].setText('{}'.format(old_value))
                return
        
        func(value)
    def save_as_default(self, who):
        self.xrdp.default.default.update({who:self.Entries[who].text()})
        self.xrdp.default.delDefault()
        self.xrdp.default.createDefault()
        
class Structures():
    """ Some default structures which the program loads. Feel free to include more!"""
    
    def __init__(self):
        self.structures = [ 'LaB6', 
                            'Si', 
                            'Diamond', 
                            'NaCl', 
                            'CsCl']
        self.structures_baseAtoms = {           'LaB6':['La','B','B','B','B','B','B'],
                                                'Si':['Si','Si','Si','Si','Si','Si','Si','Si'],
                                                'Diamond':['C','C','C','C','C','C','C','C'],
                                                'NaCl':['Cl','Cl','Cl','Cl', 'Na','Na','Na','Na'],
                                                'CsCl':['Cl', 'Cs']
                                                }
        self.structures_baseAtoms_positions = { 'LaB6':[[0,0,0],[0.1996, 0.5, 0.5],[0.5, 0.5, 0.8004],[0.5, 0.5, 0.1996],[0.5, 0.1996, 0.5],[0.5, 0.8004, 0.5],[0.8004, 0.5, 0.5]],
                                                'Si':[[0,0,0],[0.5, 0.5, 0],[0, 0.5, 0.5],[0.5, 0, 0.5],[0.25, 0.25, 0.25],[0.75, 0.75, 0.25],[0.75, 0.25, 0.75],[0.25, 0.75, 0.75]],
                                                'Diamond':[[0,0,0],[0.5, 0.5, 0],[0, 0.5, 0.5],[0.5, 0, 0.5],[0.25, 0.25, 0.25],[0.75, 0.75, 0.25],[0.75, 0.25, 0.75],[0.25, 0.75, 0.75]],
                                                'NaCl':[[0, 0, 0],[0.5, 0.5, 0],[0.5, 0, 0.5],[0, 0.5, 0.5], [0.5,0.5,0.5],[0.5, 0, 0],[0, 0.5, 0],[0, 0, 0.5]],
                                                'CsCl':[[0,0,0], [0.5,0.5,0.5]]
                                                }
        self.structures_baseAtoms_sizes = {     'LaB6':[400, 150, 150, 150, 150, 150, 150],
                                                'Si':[300,300,300,300,300,300,300,300],
                                                'Diamond':[150,150,150,150,150,150,150,150],
                                                'NaCl':[300,300,300,300,200,200,200,200],
                                                'CsCl':[300, 400]
                                                }
        self.structures_lattice_parameters = {  'LaB6':[4.155, 4.155, 4.155, 90, 90, 90],
                                                'Si':[5.43,5.43,5.43,90,90,90],
                                                'NaCl':[5.63,5.63,5.63,90,90,90],
                                                'Diamond':[3.56,3.56,3.56,90,90,90],
                                                'CsCl':[4.11,4.11,4.11,90,90,90],
                                                }
        self.structures_colors = {              'LaB6':['#ffaaaa','#00dd55','#00dd55','#00dd55','#00dd55','#00dd55','#00dd55'],
                                                'Si':['#8888dd','#8888dd','#8888dd','#8888dd','#8888dd','#8888dd','#8888dd','#8888dd'],
                                                'Diamond':['#88dd88','#88dd88','#88dd88','#88dd88','#88dd88','#88dd88','#88dd88','#88dd88'],
                                                'NaCl':['#bbbb88','#bbbb88','#bbbb88','#bbbb88','#77aadd','#77aadd','#77aadd','#77aadd'],
                                                'CsCl':['#ffddaa','#77aadd']}
        
    def getRandom(self):
        class Structure():
            def __init__(self, obj, who):
                self.structure = who
                self.baseAtoms = obj.structures_baseAtoms[who]
                self.positions = obj.structures_baseAtoms_positions[who]
                self.sizes = obj.structures_baseAtoms_sizes[who]
                self.lattice = obj.structures_lattice_parameters[who]
                self.colors = obj.structures_colors[who]
        
        a = random.choice(self.structures)
        return Structure(self, a)

class Window(QMainWindow,QWidget):
    """ Main window class"""

    # initial things
    def __init__(self, initial_values, parent=None):
        super(Window, self).__init__(parent)

        self.default = initial_values
        
        self.loadInitialParameters()
        
        self.setGeometry(10, 30, self.window_w, self.window_h)

        self.include_pXRDFigure().setGeometry(*self.pXRD_Figure_geo())
        
        self.include_toolkit()

        self.include_LatticeParams().setGeometry(*self.LatticeParams_geo())

        self.include_E().setGeometry(*self.E_geo())

        self.include_CrystalSize().setGeometry(*self.Size_geo())

        self.include_InfoFrame().setGeometry(*self.Info_geo())
        
        self.include_Params_opts().setGeometry(*self.Params_opts_geo())

        self.include_CrystalFigure().setGeometry(*self.Crystal_geo())

        self.include_BaseAtoms().setGeometry(*self.Atoms_geo())

        self.setWindowTitle("XRD Playground - version 1.0.2 2024-10-04")

        self.include_base()
        
        self.update(ul=True, xpd = True, en = True)

        self.rescale()
        
        self.update_arrow()

    def loadInitialParameters(self):
        # importing defaults
        self.inis = self.default.loadDefault()

        # loading default values from file
        self.loadDefaultValues_from_file()
        
        # things related to tth step and range
        self.Fhkl = {}
        self.calc_QiQf()
        self.create_tth_range()
        
        # things related to H K L
        self.intensity = np.zeros (len(self.tth_range)) # not sure this one...
        self.create_list_of_hkl()
        self.old_HKL_H = 1
        self.old_HKL_K = 0
        self.old_HKL_L = 0
        
        # things related to the base atoms
        self.additional_atoms = 0
        self.AddAtoms_label_At = {}
        self.AddAtoms_label_x = {}
        self.AddAtoms_label_y = {}
        self.AddAtoms_label_z = {}
        self.AddAtoms_label_pos_x = {}
        self.AddAtoms_label_pos_y = {}
        self.AddAtoms_label_pos_z = {}
        self.AddAtoms_slider_x = {}
        self.AddAtoms_slider_y = {}
        self.AddAtoms_slider_z = {}
        self.AddAtoms_entry_At = {}
        self.AddAtoms_entry_At_size = {}
        self.AddAtoms_color = {}
        self.AddAtoms_entry_pos_x = {}
        self.AddAtoms_entry_pos_y = {}
        self.AddAtoms_entry_pos_z = {}
        self.AddAtoms_groupBox = {}
        self.AddAtoms_groupBox_layout = {}        
        self.Atom_types = {}
        
        # geometry proposed
        self.ini_w = 1280
        self.ini_h = 720
        self.window_w = self.ini_w
        self.window_h = self.ini_h
        self.crystal_h = 300
        self.figure_w = 801
        self.figure_h = 497
        self.space = 5
        self.space_ini = 3

        # some app colors
        self.red_light = '#ffdddd'
        self.red = '#ff8888'
        self.red_red = '#ff0000'
        self.orange = '#ff8800'
        self.yellow = '#dddd00'
        self.green_light = '#ddffdd'
        self.green = '#88ff88'
        self.green_dark = '#66cc66'
        self.cyan = '#00bbbb'
        self.cyan_light = '#00eeee'
        self.blue = '#8888ff'
        self.blue_light = '#ddddff'
        self.blue_dark = '#6666cc'
        self.bluemagenta = '#8800ff'
        self.magenta = '#aa00aa'
        self.magenta_light = '#ee99ee'
        self.gray = '#aaaaaa'
        self.gray_light = '#dddddd'
        self.gb_bg_ini = '#dddddd'
        self.gb_bg_fin = '#ffffff'
        self.sl_col_0 = '#666666'
        self.sl_col_1 = '#999999'
        self.sl_col_2 = '#bbbbbb'
        self.sl_col_3 = '#dddddd'
        self.sl_stp_0 = '#aaaafa'
        self.sl_stp_1 = '#6666f6'
        self.crysface_color = self.cyan_light
        self.crysedge_color = self.red_red
        
        # vars needed when loading a file by the user
        self.userdata_dict = {}
        
        # colors for the atoms, when the structure is not an existing one
        self.colors = [self.gray, self.blue_dark, self.green_dark, self.orange, self.sl_col_1, self.bluemagenta, self.yellow, self.cyan, self.magenta]
        
        # taking a look if it is needed to import the lattice parameters:
        self.baseAtoms = self.inis['baseAtoms']
        self.baseAtoms_positions = self.inis['baseAtoms_positions']
        self.baseAtoms_sizes = self.inis['baseAtoms_sizes']
        if len(self.baseAtoms) != len(self.baseAtoms_positions) or len(self.baseAtoms) != len(self.baseAtoms_sizes) or self.baseAtoms == []:
            self.loadRandomStructure()
        else:
            self.loadStructureFromDefault()

        # unit cell definitions
        self.degree = np.pi/180.
        self.plotlimits = int(max([self.par0['a'], self.par0['b'],self.par0['c']])) + 1
        self.plotlimits_min = int(min([self.par_min['a'], self.par_min['b'],self.par_min['c']])) + 1
        self.plotlimits_max = int(max([self.par_max['a'], self.par_max['b'],self.par_max['c']])) + 1   
        
    def loadDefaultValues_from_file(self):
        # language, font and size used
        self.l = Language()
        self.l.setLanguage(self.inis['language'])
        self.font = self.inis['font_ini']
        self.fontsize = int(float(self.inis['fontsize_ini']))
        # initial energy
        self.init_E = float(self.inis['E_ini'])
        self.E = self.init_E
        xu.energy(self.init_E*1000)
        self.E_min = float(self.inis['E_min'])
        self.E_max = float(self.inis['E_max'])
        # initial crystallite size
        self.init_D = float(self.inis['CrysSize_ini'])
        self.D = self.init_D
        self.D_min = float(self.inis['CrysSize_min'])
        self.D_max = float(self.inis['CrysSize_max'])
        # initial lattice parameters boundaries
        self.LatticeParams = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        self.par_name = {'a': 'a',   'b':'b',   'c':'c',   'alpha':'\u03b1', 'beta':'\u03b2', 'gamma':'\u03b3'}
        self.par_button_clicks = {'a':self.button_a, 'b':self.button_b, 'c':self.button_c, 'alpha':self.button_alpha, 'beta':self.button_beta, 'gamma':self.button_gamma}
        self.par_slider_change = {'a':self.slider_a, 'b':self.slider_b, 'c':self.slider_c, 'alpha':self.slider_alpha, 'beta':self.slider_beta, 'gamma':self.slider_gamma}
        self.par_min =  {'a': float(self.inis['a_min']),  'b': float(self.inis['b_min']),  'c':float(self.inis['c_min']),  'alpha':float(self.inis['alpha_min']),  'beta':float(self.inis['beta_min']),  'gamma':float(self.inis['gamma_min'])}
        self.par_max =  {'a': float(self.inis['a_max']),  'b': float(self.inis['b_max']),  'c':float(self.inis['c_max']),  'alpha':float(self.inis['alpha_max']),  'beta':float(self.inis['beta_max']),  'gamma':float(self.inis['gamma_max'])}
        # initial tth step and range
        self.tth_min = float(self.inis['tth_min'])
        self.tth_max = float(self.inis['tth_max'])
        self.tth_step = float(self.inis['tth_step'])
        # initial H K L maximum values
        self.h_max = int(float(self.inis['h_max']))
        self.k_max = int(float(self.inis['k_max']))
        self.l_max = int(float(self.inis['l_max']))
        
    def loadRandomStructure(self):
        '''This function loads a random structure from a pool when opening the program'''
        self.structure = Structures().getRandom()
        self.color_init = self.structure.colors
        self.pos_x_init = np.array(self.structure.positions).T[0]
        self.pos_y_init = np.array(self.structure.positions).T[1]
        self.pos_z_init = np.array(self.structure.positions).T[2]
        self.baseAtoms = self.structure.baseAtoms
        self.sizes_init = self.structure.sizes
        self.par0 = {   'a': self.structure.lattice[0],
                        'b': self.structure.lattice[1],
                        'c': self.structure.lattice[2],
                        'alpha':self.structure.lattice[3],
                        'beta': self.structure.lattice[4],
                        'gamma':self.structure.lattice[5]}
        self.Atom_types.update({0:self.baseAtoms[0]})
        self.color_atom0 = self.color_init[0]
        
    def loadStructureFromDefault(self):
        """ This fucntion was necessary in the past, but I believe it is not anymore. It was substituted by the loading of the random structure """
        self.Atom_types.update({0:self.baseAtoms[0]})
        self.color_init = self.colors
        self.pos_x_init = self.baseAtoms_positions.T[0]
        self.pos_y_init = self.baseAtoms_positions.T[1]
        self.pos_z_init = self.baseAtoms_positions.T[2]
        self.sizes_init = self.baseAtoms_sizes
        self.last_atom = self.baseAtoms[-1]
        self.last_atom_size = self.baseAtoms_sizes[-1]
        self.par0 = {   'a': float(self.inis['a_ini']),  
                        'b': float(self.inis['b_ini']),  
                        'c':float(self.inis['c_ini']),  
                        'alpha':float(self.inis['alpha_ini']),  
                        'beta':float(self.inis['beta_ini']),  
                        'gamma':float(self.inis['gamma_ini'])}
        self.color_atom0 = self.blue


    # lists and updates
    def calc_QiQf(self):
        """Calculates the Q limits depending on the tth and energy """
        self.Qi = 4*np.pi*self.E/12.398*np.sin(self.tth_min*np.pi/360.)
        self.Qf = 4*np.pi*self.E/12.398*np.sin(self.tth_max*np.pi/360.)
        
    def create_tth_range(self):  
        """Just create a tth range based on tth minimum and maximum"""
        self.tth_range =  np.arange (self.tth_min,self.tth_max,self.tth_step)
        
    def create_list_of_hkl(self):
        """ creates a list of HKLs based on the maximum values for them. """
        self.list_of_hkl = []
        for h in range (-self.h_max,self.h_max+1):
            for k in range (-self.k_max,self.k_max+1):
                for l in range (-self.l_max,self.l_max+1):
                    if not ( h == 0 and k == 0 and l == 0): self.list_of_hkl.append([h,k,l])



    # geometry proposed and calculations based on resize event
    def resizeEvent(self, event):
        self.calculate_geometry()
        QMainWindow.resizeEvent(self, event)
        
    def pXRD_Figure_geo(self):
        """ geometry for the main figure"""
        x = self.space
        y = self.space_ini
        w = int(self.figure_w/float(self.ini_w)*self.window_w)
        h = int(float(self.figure_h)/self.ini_h*self.window_h)
        return [x,y,w,h]
        
    def LatticeParams_geo(self):
        """ geometry for the lattice parameters region"""
        x = self.space
        y = int(float(self.figure_h)/self.ini_h*self.window_h)+self.space+self.space_ini
        w = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2)
        h = self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space
        return [x,y,w,h]
        
    def E_geo(self):
        """ geometry for energu and wavelength widget"""
        x = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2) + 2*self.space
        y = int(float(self.figure_h)/self.ini_h*self.window_h)+self.space+self.space_ini
        w = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2)
        h = int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*2.5/6)
        return [x,y,w,h]
        
    def Size_geo(self):
        """geometry for the cristallite size widget"""
        x = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2) + 2*self.space
        y = int(float(self.figure_h)/self.ini_h*self.window_h)+self.space+self.space_ini + int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*2.5/6) + self.space
        w = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2)
        h = int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*1.5/6)
        return [x,y,w,h]
        
    def Info_geo(self):
        """ geometry for the information widget. """
        x = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2) + 2*self.space + 1*self.space + 50
        a = int(float(self.figure_h)/self.ini_h*self.window_h)+self.space+self.space_ini + int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*2.5/6) + self.space
        b = int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*1.5/6)+ self.space
        y = a + b
        w = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2) - 50 - 1*self.space
        h = int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*2/6)
        return [x,y,w,h]
        
    def Params_opts_geo(self):
        """ geometry for the setting button widget. """
        x = int((int(self.figure_w/float(self.ini_w)*self.window_w)-self.space)/2) + 2*self.space 
        a = int(float(self.figure_h)/self.ini_h*self.window_h)+self.space+self.space_ini + int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*2.5/6) + self.space
        b = int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*1.5/6)+ self.space
        y = a + b
        w = 50
        h = int((self.window_h-int(float(self.figure_h)/self.ini_h*self.window_h)-self.space_ini-2*self.space-3*self.space)*2/6)
        return [x,y,w,h]
        
    def Crystal_geo(self):
        """ geometry for the crystal figure widget """
        x = int(self.figure_w/float(self.ini_w)*self.window_w)+2*self.space
        y = self.space_ini
        w = self.window_w-x-self.space
        h = int(float(self.crystal_h)/self.ini_h*self.window_h)
        return [x,y,w,h]
        
    def Atoms_geo(self):
        """ geometry for the atoms widget"""
        x = int(self.figure_w/float(self.ini_w)*self.window_w)+2*self.space
        y = int(float(self.crystal_h)/self.ini_h*self.window_h) + self.space_ini+self.space
        w = self.window_w-x-self.space
        h = self.window_h-int(float(self.crystal_h)/self.ini_h*self.window_h)-2*self.space-self.space_ini
        return [x,y,w,h]
        
    def calculate_geometry(self):
        """ gets the size of the window to calculate the new size for the widgets. Much better than the automatic."""
        self.window_w = self.width()
        self.window_h = self.height()
        self.space = 5
        self.space_ini = 3
        self.pXRDFigure_groupBox.setGeometry(*self.pXRD_Figure_geo())
        self.LatticeParams_groupBox.setGeometry(*self.LatticeParams_geo())
        self.E_groupBox.setGeometry(*self.E_geo())
        self.CrystalSize_groupBox.setGeometry(*self.Size_geo())
        self.CrystalFigure_groupBox.setGeometry(*self.Crystal_geo())
        self.BaseAtoms_groupBox.setGeometry(*self.Atoms_geo())
        self.Info_groupBox.setGeometry(*self.Info_geo())
        self.Params_opts_groupBox.setGeometry(*self.Params_opts_geo())




    def eventFilter(self, source, event, flag = 0):
        """ event filter, but I did not finish this. Don't believe will. Probably remove from next versions. """
        if flag == 0:
            if event.type() == QEvent.Enter and source is self.pXRDFigure_groupBox: self.Info_label.setText(self.l.what['xpd'])
        elif flag in self.LatticeParams:
            if event.type() == QEvent.Enter and source is self.LatticeParams_button[flag]: self.Info_label.setText(self.l.what['lattpar'])
            elif event.type() == QEvent.Enter and source is self.LatticeParams_slider[flag]: self.Info_label.setText(self.l.what['lattpar'])
            elif event.type() == QEvent.Enter and source is self.LatticeParams_entry[flag]: self.Info_label.setText(self.l.what['lattpar'])
        
        if event.type() == QEvent.Leave: self.Info_label.setText('')
        return super(Window, self).eventFilter(source, event)

    # main graph functions and options
    def include_pXRDFigure(self):
        """ creating the powder XRD figure and other things inside that groupbox"""
        # just the name of the group
        self.pXRDFigure_groupBox = QGroupBox(self.l.what['title'], self)
        self.pXRDFigure_groupBox.setFont(QFont(self.font,self.fontsize))
        group_opts = self.GroupBox_StyleSheet(self.green, self.green_light)
        self.pXRDFigure_groupBox.setStyleSheet(group_opts)
        self.pXRDFigure_groupBox.installEventFilter(self)
        
        # matplotlib figure and ax.
        self.XPDFigure, self.pXRDax = plt.subplots()
        plt.subplots_adjust(left=0.08,right=0.98, bottom=0.08, top=0.98)
        
        self.main_plot, = self.pXRDax.plot (self.tth_range,self.intensity, 'ko-',lw = 2.5, markersize = 4)
        self.colored_plots = []
        for i in ['r','g','b']:
            a, = self.pXRDax.plot (self.tth_range,self.intensity, '{}o-'.format(i),lw = 1.2, markersize = 2)
            a.set_visible(False)
            a.scale = 1.
            self.colored_plots.append(a)
            
        self.define_pXRDax_limits()
        #self.pXRDax.set_xlim(self.tth_min-5,self.tth_max+5)
        
        self.hkl_text = self.pXRDax.text(5, 5, '')
        self.line, = self.pXRDax.plot([0,0],[0,0], 'r--', lw = 1)
        
        # Canvas Widget that displays the `figure`; it takes the `figure` instance as a parameter to its __init__
        self.XPDcanvas = FigureCanvas(self.XPDFigure) 
        
        # Navigation widget; it takes the Canvas widget and a parent
        self.tool_manager = ToolManager(self.XPDFigure)
        self.XPDtoolbar = ToolbarQt(self.tool_manager, self.pXRDFigure_groupBox) 
        

        
        # Setting the layout:
        Figure_layout = QVBoxLayout()
        Figure_layout.addSpacing(5)
        Figure_layout.addWidget(self.XPDtoolbar)
        Figure_layout.addWidget(self.XPDcanvas)
        self.pXRDFigure_groupBox.setLayout(Figure_layout)
        
        return self.pXRDFigure_groupBox

    def define_pXRDax_limits(self):
        """ set the ax limits"""
        self.pXRDax.set_xlim(self.tth_min-5,self.tth_max+5)
        
    def include_toolkit(self):
        """ include different things on the tookit """
        backend_tools.add_tools_to_manager(self.tool_manager)
        backend_tools.add_tools_to_container(self.XPDtoolbar)
        self.tool_manager.remove_tool('forward') #just to get space
        self.tool_manager.remove_tool('back')    
        
        
        for i in range(3):
            self.tool_manager.add_tool('Freeze_{}'.format(i), Freeze, curve=i, plot = self.colored_plots[i], main_plot = self.main_plot, description = self.l.what)
            self.XPDtoolbar.add_tool('Freeze_{}'.format(i),'my')
        self.tool_manager.add_tool('Rescale', Rescale_y, func = self.rescale, main_plot = self.main_plot, description = self.l.what['rescale'])
        self.XPDtoolbar.add_tool('Rescale','my', 0)
        self.tool_manager.add_tool('LoadData', Load_Data, xrdp = self, description = self.l.what['loaddata'])
        self.XPDtoolbar.add_tool('LoadData','mydata', 0)
        self.tool_manager.add_tool('Settings', Settings_Data, xrdp = self, description = self.l.what['settings'], func = self.settings_user_data_preparation)
        self.XPDtoolbar.add_tool('Settings','mydata', 0)
        
        self.include_HKL()
        self.include_settings_graph()
        
    def include_HKL(self): 
        """ H, K and L label and edit boxes at the end of toolbar"""
        self.text_H = QLabel("  H ")
        self.text_K = QLabel("  K ")
        self.text_L = QLabel("  L ")
        self.text_H.setStyleSheet('QLabel {font: bold '+'{}px'.format(self.fontsize)+'}')
        self.text_K.setStyleSheet('QLabel {font: bold '+'{}px'.format(self.fontsize)+'}')
        self.text_L.setStyleSheet('QLabel {font: bold '+'{}px'.format(self.fontsize)+'}')

        self.le_H = QLineEdit(str(self.old_HKL_H))
        self.le_K = QLineEdit(str(self.old_HKL_K))
        self.le_L = QLineEdit(str(self.old_HKL_L))
        self.le_H.setMaxLength(2)
        self.le_K.setMaxLength(2)
        self.le_L.setMaxLength(2)
        self.le_H.setAlignment(Qt.AlignCenter)
        self.le_K.setAlignment(Qt.AlignCenter)
        self.le_L.setAlignment(Qt.AlignCenter)
        entry_opts = self.LEtool_StyleSheet(self.magenta_light)
        self.le_H.setStyleSheet(entry_opts)
        self.le_K.setStyleSheet(entry_opts)
        self.le_L.setStyleSheet(entry_opts)
        self.le_H.editingFinished.connect(self.update_HKL_H)
        self.le_K.editingFinished.connect(self.update_HKL_K)
        self.le_L.editingFinished.connect(self.update_HKL_L)
        self.le_H.setToolTip(self.l.what['hkl_inf'] + ' ' + self.l.what['hkl_inf2'].format(max = self.h_max))
        self.le_K.setToolTip(self.l.what['hkl_inf'] + ' ' + self.l.what['hkl_inf2'].format(max = self.k_max))
        self.le_L.setToolTip(self.l.what['hkl_inf'] + ' ' + self.l.what['hkl_inf2'].format(max = self.l_max))

        self.showHKL_check = QCheckBox('')
        check_opts = self.Checkbox_StyleSheet(color = self.magenta_light)
        self.showHKL_check.setStyleSheet(check_opts)
        self.showHKL_check.setChecked(False)
        self.showHKL_check.stateChanged.connect(self.check_showHKL_TF)
        self.showHKL_check.setToolTip(self.l.what['showHKL'])

        self.XPDtoolbar.addWidget(self.showHKL_check)
        self.XPDtoolbar.addWidget(self.text_H)
        self.XPDtoolbar.addWidget(self.le_H)
        self.XPDtoolbar.addWidget(self.text_K)
        self.XPDtoolbar.addWidget(self.le_K)
        self.XPDtoolbar.addWidget(self.text_L)
        self.XPDtoolbar.addWidget(self.le_L)
        
    def include_settings_graph(self):
        """include the setting button at the very end of the toolbar """
        self.settings_graph_button = QPushButton('')
        self.settings_graph_button.clicked.connect(self.settings_graph)
        button_opts = self.PushButton_Toolbox_StyleSheet(self.sl_col_1)
        self.settings_graph_button.setStyleSheet(button_opts)
        self.settings_graph_button.setToolTip(self.l.what['graph_settings'])
        icon = qta.icon('ri.settings-5-fill', color='k', scale_factor = 1.3)
        self.settings_graph_button.setIcon(icon)
        self.XPDtoolbar.addSeparator()
        self.XPDtoolbar.addWidget(self.settings_graph_button)
        
    def update_HKL_H(self):
        """ callback function when updating H edit box at the toobar"""
        val = self.le_H.text()
        try:
            val_ = int(val)
            if val_ <-9 or val_ > 9:

                text = '"{}" {} {}'.format(val_, self.l.what['problem_a'], 'H')
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_H.setText(str(self.old_HKL_H))
            elif val_ == 0 and int(self.le_K.text()) == 0 and int(self.le_L.text()) == 0:
                text = '{}'.format(self.l.what['hkl_000'])
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_H.setText(str(self.old_HKL_H))

            else:
                self.old_HKL_H = val_
                self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, self.l.what['problem_a'], 'H')
            self.Update_Info_label(text = text, bkg = self.red)
            self.le_H.setText(str(self.old_HKL_H))

    def update_HKL_K(self):
        """ callback function when updating K edit box at the toobar"""
        val = self.le_K.text()
        try:
            val_ = int(val)
            if val_ <-9 or val_ > 9:
                text = '"{}" {} {}'.format(val_, self.l.what['problem_a'], 'K')
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_K.setText(str(self.old_HKL_K))
            elif val_ == 0 and int(self.le_H.text()) == 0 and int(self.le_L.text()) == 0:
                text = '{}'.format(self.l.what['hkl_000'])
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_K.setText(str(self.old_HKL_K))
            else:
                self.old_HKL_K = val_
                self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, self.l.what['problem_a'], 'K')
            self.Update_Info_label(text = text, bkg = self.red)
            self.le_K.setText(str(self.old_HKL_K))

    def update_HKL_L(self):
        """ callback function when updating L edit box at the toobar"""
        val = self.le_L.text()
        try:
            val_ = int(val)
            if val_ <-9 or val_ > 9:
                text = '"{}" {} {}'.format(val_, self.l.what['problem_a'], 'L')
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_L.setText(str(self.old_HKL_L))
            elif val_ == 0 and int(self.le_H.text()) == 0 and int(self.le_K.text()) == 0:
                text = '{}'.format(self.l.what['hkl_000'])
                self.Update_Info_label(text = text, bkg = self.red)
                self.le_L.setText(str(self.old_HKL_L))
            else:
                self.old_HKL_L = val_
                self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, self.l.what['problem_a'],'L')
            self.Update_Info_label(text = text, bkg = self.red)
            self.le_L.setText(str(self.old_HKL_L))

    def update_arrow(self):
        """ Function which updates the arrow taht indicates the position of the HKL peak at the main graph"""
        h = self.old_HKL_H
        k = self.old_HKL_K
        l = self.old_HKL_L
        Q = self.Qhkl(h, k, l)
        wvl = self.Wvl_slider.value()
        _2theta = self.Q2tth (Q, wvl)
        ymax = self.pXRDax.get_ylim()[1]
        try: 
            self.arrow_.remove()
        except:
            pass
        arrow = Arrow(_2theta,ymax*0.93,0,-ymax*0.09, color="#aa0088")
        if self.showHKL_check.isChecked():
            self.arrow_ = self.pXRDax.add_patch(arrow)
            self.line.set_data([_2theta, _2theta], [0, ymax*0.77])
            self.hkl_text.set_text('{} {} {}'.format(h, k, l))
            self.hkl_text.set_position((_2theta, ymax*0.95))
        else:
            self.line.set_data([0,0],[0,0])
            self.hkl_text.set_text('')

    def check_showHKL_TF(self):
        """ check if the HKL arrow and plane will be shown or not """
        self.update(ul=False, xpd = True, en = False)
        if self.showHKL_check.isChecked(): self.calc_HKL_planes()
        
    def settings_graph(self):
        """ function that opens a window where the user can change some parameters and default values for future uses of the application"""
        self.graph_options = { 'tth_min':self.tth_min, 
                    'tth_max':self.tth_max, 
                    'tth_step':self.tth_step,
                    'h_max': self.h_max,
                    'k_max': self.k_max,
                    'l_max': self.l_max}
        self.graph_functions = {   'tth_min':  self.set_tth_min,
                        'tth_max':  self.set_tth_max,
                        'tth_step': self.set_tth_step,
                        'h_max':    self.set_h_max,
                        'k_max':    self.set_k_max,
                        'l_max':    self.set_l_max}
        self.graph_names = {   'tth_min':'2\u03b8 min', 
                    'tth_max':'2\u03b8 max', 
                    'tth_step':'2\u03b8 step',
                    'h_max': 'H max ',
                    'k_max': 'K max ',
                    'l_max': 'L max '}
        self.graph_tooltips = {'tth_min':"minimum 2\u03b8 value for calculation", 
                    'tth_max':"maximum 2\u03b8 value for calculation", 
                    'tth_step':"2\u03b8 step (warning: increases calc time)",
                    'h_max': "\u00b1H for peak calculation (warning: increases calc time a lot!)",
                    'k_max': "\u00b1K for peak calculation (warning: increases calc time a lot!)",
                    'l_max': "\u00b1L for peak calculation (warning: increases calc time a lot!)"}
        self.graph_types = { 'tth_min':'float', 
                    'tth_max':'float',
                    'tth_step':'float',
                    'h_max': 'int',
                    'k_max': 'int',
                    'l_max': 'int'}
        graph_min_lims = {  'tth_min':0, 
                            'tth_max':self.tth_min,
                            'tth_step':0,
                            'h_max': 0,
                            'k_max': 0,
                            'l_max': 0}
        graph_max_lims = {  'tth_min':self.tth_max, 
                            'tth_max':180,
                            'tth_step':1,
                            'h_max': 20,
                            'k_max': 20,
                            'l_max': 20}
        color = {           'tth_min':self.magenta_light, 
                            'tth_max':self.magenta_light,
                            'tth_step':self.magenta_light,
                            'h_max':self.magenta_light,
                            'k_max':self.magenta_light,
                            'l_max':self.magenta_light 
                            }
        save =              {'tth_min':True, 
                            'tth_max':True,
                            'tth_step':True,
                            'h_max':True,
                            'k_max':True,
                            'l_max':True
                            }
        title = "Graph Options"
        self.w1 = PopUpOpt(self, self.graph_options, self.graph_functions, self.graph_names, self.graph_tooltips, self.graph_types, graph_min_lims, graph_max_lims, color, title, save)
        self.w1.setGeometry(QRect(100, 100, 200, 200))
        self.w1.show()

    def set_tth_min(self, tth_min):
        """ function that changes the minimum value for tth, and associated parameters"""
        self.tth_min = tth_min
        self.create_tth_range()
        self.calc_QiQf()
        self.update(ul=True, from_opts = True)
        self.define_pXRDax_limits()
        return self.tth_min
        
    def set_tth_max(self, tth_max):
        """ function that changes the maximum value for tth, and associated parameters"""
        self.tth_max = tth_max
        self.create_tth_range()
        self.calc_QiQf()
        self.update(ul=True, from_opts = True)
        self.define_pXRDax_limits()
        return self.tth_max
        
    def set_tth_step(self, tth_step):
        """ function that changes the atep for tth, and associated parameters"""
        self.tth_step = tth_step
        self.create_tth_range()
        self.update(ul=True, from_opts = True)
        return self.tth_step
        
    def set_h_max (self, h_max):
        """ set the maximum value for H used in the simulations """
        self.h_max = h_max
        self.create_list_of_hkl()
        self.update(ul=True)
        return self.h_max
        
    def set_k_max (self, k_max):
        """ set the maximum value for K used in the simulations """
        self.k_max = k_max
        self.create_list_of_hkl()
        self.update(ul=True)
        return self.k_max
        
    def set_l_max (self, l_max):
        """ set the maximum value for L used in the simulations """
        self.l_max = l_max
        self.create_list_of_hkl()
        self.update(ul=True)
        return self.l_max

    def settings_user_data_preparation(self):
        """this function helps to open the dialog for including the user data  """
        noc = len(list(self.userdata_dict.keys()))
        labels = []
        colors = []
        dotsize = []

        if noc != 0:
            for i in list(self.userdata_dict.keys()):
                self.userdata_dict[i].update({'size':self.userdata_dict[i]['scatters'].get_sizes()[0]})
                self.userdata_dict[i].update({'label':i, 'color': rgb2hex(self.userdata_dict[i]['scatters'].get_facecolors()[0].tolist())})
                
        return noc, self.userdata_dict, self.updating_user_data 

    def updating_user_data(self, i, label):
        """ this function helps to update used data when something is changed"""
        self.userdata_dict[label]['scatters'].set_offsets(np.c_[self.userdata_dict[label]['datax'],self.userdata_dict[label]['datay']*self.userdata_dict[label]['scale']])
        sizes = np.ones(len(self.userdata_dict[label]['datax']))*self.userdata_dict[label]['size']
        self.userdata_dict[label]['scatters'].set_sizes(sizes)
        self.XPDcanvas.draw_idle()
    
       
    # lattice parameters, energy and crystallite size functions and options
    def include_LatticeParams(self):
        """ including the group box of the lattice parameter """
        # just the name of the group
        self.LatticeParams_groupBox = QGroupBox(self.l.what['pars'], self)
        self.LatticeParams_groupBox.setFont(QFont(self.font,self.fontsize))
        group_opts = self.GroupBox_StyleSheet(self.green, self.green_light)
        self.LatticeParams_groupBox.setStyleSheet(group_opts)
        #self.LatticeParams_groupBox.installEventFilter(self)
        
        
        #include buttons for restart, sliders and labels 
        self. LatticeParams_button = {}
        button_opts = self.PushButton_StyleSheet(self.green)

        self. LatticeParams_slider = {}
        slider_opts = self.Slider_StyleSheet()
        
        self. LatticeParams_entry = {}
        entry_opts = self.LineEdit_StyleSheet(self.green)

        for i in self. LatticeParams:
            self.LatticeParams_button.update({i:QPushButton(self.par_name[i])})
            self.LatticeParams_button[i].setFont(QFont(self.font,self.fontsize))
            self.LatticeParams_button[i].clicked.connect(self.par_button_clicks[i])
            self.LatticeParams_button[i].setStyleSheet(button_opts)
            self.LatticeParams_button[i].setToolTip(self.par_name[i] + ': ' + self.l.what['tt_b_pars'])
            if i in ['a', 'b', 'c']:
                self. LatticeParams_slider.update({i:DoubleSlider(3, Qt.Horizontal)})
            else:
                self. LatticeParams_slider.update({i:DoubleSlider(1, Qt.Horizontal)})
            self.LatticeParams_slider[i].setStyleSheet(slider_opts)
            self. LatticeParams_slider[i].setMinimum(self.par_min[i])
            self. LatticeParams_slider[i].setMaximum(self.par_max[i])
            self. LatticeParams_slider[i].setValue(self.par0[i])
            self. LatticeParams_slider[i].setSingleStep(1)
            self. LatticeParams_slider[i].setTickPosition(QSlider.NoTicks)
            self. LatticeParams_slider[i].valueChanged.connect(self.par_slider_change[i])
            self. LatticeParams_slider[i].setToolTip(self.l.what['tt_s_pars'])

            self.LatticeParams_entry.update({i:QLineEdit(str(self.par0[i]))})
            self.LatticeParams_entry[i].setStyleSheet(entry_opts)
            self.LatticeParams_entry[i].setFont(QFont(self.font,self.fontsize))
            self.LatticeParams_entry[i].setMaxLength(5)
            self.LatticeParams_entry[i].editingFinished.connect(lambda par = i: self.update_sliders(par))
            self.LatticeParams_entry[i].setFixedWidth(60)
            self.LatticeParams_entry[i].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.LatticeParams_entry[i].setToolTip(self.l.what['tt_e_pars'])

        LatticeParams_layout = QGridLayout()
        for i in range(6):
             LatticeParams_layout.addWidget(self. LatticeParams_button[self. LatticeParams[i]], i, 21)
             LatticeParams_layout.addWidget(self. LatticeParams_slider[self. LatticeParams[i]], i, 22, 1, 6)
             LatticeParams_layout.addWidget(self. LatticeParams_entry[self. LatticeParams[i]], i, 29)
        self.LatticeParams_groupBox.setLayout( LatticeParams_layout)
        
        return self.LatticeParams_groupBox

    def button_a (self): 
        """ function called when button a is clicked"""
        self. LatticeParams_slider['a'].setValue(self.par0['a'])
        self. LatticeParams_entry['a'].setText(str(self. LatticeParams_slider['a'].value()))
        self.update(ul=True, en = False)
        
    def button_b (self): 
        """ function called when button B is clicked"""
        self. LatticeParams_slider['b'].setValue(self.par0['b'])
        self. LatticeParams_entry['b'].setText(str(self. LatticeParams_slider['b'].value()))
        self.update(ul=True, en = False)
        
    def button_c (self): 
        """ function called when button c is clicked"""
        self. LatticeParams_slider['c'].setValue(self.par0['c'])
        self. LatticeParams_entry['c'].setText(str(self. LatticeParams_slider['c'].value()))
        self.update(ul=True, en = False)
        
    def button_alpha (self): 
        """ function called when button alpha is clicked"""
        self. LatticeParams_slider['alpha'].setValue(self.par0['alpha'])
        self. LatticeParams_entry['alpha'].setText(str(self. LatticeParams_slider['alpha'].value()))
        self.update(ul=True, en = False)
        
    def button_beta  (self): 
        """ function called when button beta is clicked"""
        self. LatticeParams_slider['beta'].setValue(self.par0['beta'])
        self. LatticeParams_entry['beta'].setText(str(self. LatticeParams_slider['beta'].value()))
        self.update(ul=True, en = False)
        
    def button_gamma (self): 
        """ function called when button gamma is clicked"""
        self. LatticeParams_slider['gamma'].setValue(self.par0['gamma'])
        self. LatticeParams_entry['gamma'].setText(str(self. LatticeParams_slider['gamma'].value()))
        self.update(ul=True, en = False)
        
    def slider_a (self): 
        """ function called when slider a is changed"""
        self.LatticeParams_entry['a'].setText(str(self.LatticeParams_slider['a'].value()))
        self.update(ul=True, en = False)
        
    def slider_b (self): 
        """ function called when slider b is changed"""
        self.LatticeParams_entry['b'].setText(str(self. LatticeParams_slider['b'].value()))
        self.update(ul=True, en = False)
        
    def slider_c (self): 
        """ function called when slider c is changed"""
        self.LatticeParams_entry['c'].setText(str(self. LatticeParams_slider['c'].value()))
        self.update(ul=True, en = False)
        
    def slider_alpha (self): 
        """ function called when slider alpha is changed"""
        self.LatticeParams_entry['alpha'].setText(str(self. LatticeParams_slider['alpha'].value()))
        self.update(ul=True, en = False)
        
    def slider_beta  (self): 
        """ function called when slider beta is changed"""
        self.LatticeParams_entry['beta'].setText(str(self. LatticeParams_slider['beta'].value()))
        self.update(ul=True, en = False)
        
    def slider_gamma (self): 
        """ function called when slider gamma is changed"""
        self.LatticeParams_entry['gamma'].setText(str(self. LatticeParams_slider['gamma'].value()))
        self.update(ul=True, en = False)
        
    def update_sliders(self, par): 
        """ function called when edit boxes texts are changed"""
        val = self.LatticeParams_entry[par].text()
        try:
            self.LatticeParams_slider[par].setValue(float(val))
            self.update(ul=True, en = False)
        except:
            text = '"{}" {} {}'.format(val, self.l.what['problem_a'], self.l.what[par])
            self.Update_Info_label(text = text, bkg = self.red)
            self.LatticeParams_entry[par].setText(str(self.LatticeParams_slider[par].value()))

    def include_E(self):
        """ including the group box of the energy and wavelength """

        # just the name of the group
        self.E_groupBox = QGroupBox(self.l.what['wvl'], self)
        self.E_groupBox.setFont(QFont(self.font,self.fontsize))
        group_opts = self.GroupBox_StyleSheet(self.red, self.red_light)
        self.E_groupBox.setStyleSheet(group_opts)
        
        self.E_button = QPushButton('E (keV)')
        self.E_button.clicked.connect(self.reset_E)
        button_opts = self.PushButton_StyleSheet(self.red)
        self.E_button.setStyleSheet(button_opts)
        self.E_button.setToolTip(self.l.what['en_0'] + ': ' + self.l.what['tt_b_pars'])
        
        self.Wvl_button = QPushButton('\u03bb (\u212b)')
        self.Wvl_button.clicked.connect(self.reset_E)
        self.Wvl_button.setStyleSheet(button_opts)
        self.Wvl_button.setToolTip(self.l.what['wvl_0'] + ': ' + self.l.what['tt_b_pars'])
        
        self.E_slider = DoubleSlider(4,Qt.Horizontal)
        slider_opts = self.Slider_StyleSheet()
        self.E_slider.setStyleSheet(slider_opts)
        self.E_slider.setMinimum(self.E_min)
        self.E_slider.setMaximum(self.E_max)
        self.E_slider.setValue(self.E)
        self.E_slider.setTickPosition(QSlider.NoTicks)
        self.E_slider.valueChanged.connect(self.E_slider_change)
        self.E_slider.setToolTip(self.l.what['tt_s_pars'])

        self.Wvl_slider = DoubleSlider(4,Qt.Horizontal)
        self.Wvl_slider.setStyleSheet(slider_opts)
        self.Wvl_slider.setMinimum(12.398/self.E_max)
        self.Wvl_slider.setMaximum(12.398/self.E_min)
        self.Wvl_slider.setValue(12.398/self.E)
        self.Wvl_slider.setTickPosition(QSlider.NoTicks)
        self.Wvl_slider.valueChanged.connect(self.Wvl_slider_change)
        self.Wvl_slider.setToolTip(self.l.what['tt_s_pars'])

        self.E_entry = QLineEdit()
        entry_opts = self.LineEdit_StyleSheet(self.red)

        self.E_entry.setFont(QFont(self.font,self.fontsize))
        self.E_entry.setStyleSheet(entry_opts)
        self.E_entry.setText('{: 5.2f}'.format(self.E))
        self.E_entry.setMaxLength(5)
        self.E_entry.setFixedWidth(60)
        self.E_entry.setToolTip(self.l.what['tt_e_pars'])

        self.E_entry.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.E_entry.editingFinished.connect(self.E_update_slider)

        self.Wvl_entry = QLineEdit()
        self.Wvl_entry.setFont(QFont(self.font,self.fontsize))
        self.Wvl_entry.setStyleSheet(entry_opts)
        self.Wvl_entry.setText('{: 5.2f}'.format(12.398/self.E))
        self.Wvl_entry.setMaxLength(5)
        self.Wvl_entry.setFixedWidth(60)
        self.Wvl_entry.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.Wvl_entry.editingFinished.connect(self.Wvl_upadate_slider)
        self.Wvl_entry.setToolTip(self.l.what['tt_e_pars'])

        E_layout = QGridLayout()
        E_layout.addWidget(self.E_button, 0, 0)
        E_layout.addWidget(self.E_slider, 0, 1, 1, 5)
        E_layout.addWidget(self.E_entry, 0, 6)
        E_layout.addWidget(self.Wvl_button, 1, 0)
        E_layout.addWidget(self.Wvl_slider, 1, 1, 1, 5)
        E_layout.addWidget(self.Wvl_entry, 1, 6)
        
        self.E_groupBox.setLayout(E_layout)
        
        return self.E_groupBox

    def reset_E(self):
        """ function called when clicked the energy button, in order to return to initial value"""
        self.E = self.init_E
        self.E_slider.setValue(self.E)
        self.Wvl_slider.setValue(12.398/self.E)
        self.updateQs()
        self.update(ul=True, en = True)

    def E_slider_change(self):
        """ function called when the energy slider is changed"""
        self.E = self.E_slider.value()
        self.E_entry.setText('{: 5.2f}'.format(self.E))
        self.Wvl_entry.setText('{: 5.2f}'.format(12.398/self.E))
        self.Wvl_slider.setValue(12.398/self.E)
        self.update(ul=True, en = True)

    def Wvl_slider_change(self):
        """ function called when the wavelength slider is changed"""
        self.Wvl = self.Wvl_slider.value()
        self.Wvl_entry.setText('{: 5.2f}'.format(12.398/self.E))
        self.E_entry.setText('{: 5.2f}'.format(self.E))
        self.E_slider.setValue(12.398/self.Wvl)
        self.updateQs()
        self.update(ul=True, en = True)

    def E_update_slider(self):
        """function called when entering a new value in energy text box """
        val = self.E_entry.text()
        try:
            self.E = float(val)
            self.E_slider.setValue(self.E)
            self.Wvl_slider.setValue(12.398/self.E)
            self.Wvl_entry.setText('{: 5.2f}'.format(12.398/self.E))
            self.updateQs()
            self.update(ul=True, en = True)
        except:
            text = '"{}" {} {}'.format(val, self.l.what['problem_a'], self.l.what['en_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.E_entry.setText('{: 5.2f}'.format(self.E))

    def Wvl_upadate_slider(self):
        """function called when entering a new value in wavelength text box """
        val = self.Wvl_entry.text()
        try:
            self.E = 12.398/float(val)
            self.E_slider.setValue(self.E)
            self.Wvl_slider.setValue(float(self.Wvl_entry.text()))
            self.E_entry.setText('{: 5.2f}'.format(self.E))
            self.updateQs()
            self.update(ul=True, en = True)
        except:
            text = '"{}" {} {}'.format(val, self.l.what['problem_a'], self.l.what['wvl_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.Wvl_entry.setText('{: 5.2f}'.format(12.398/self.E))

    def include_CrystalSize(self):
        """ including the group box of the crystal size parameter """
        # just the name of the group
        self.CrystalSize_groupBox = QGroupBox(self.l.what['size'], self) 
        self.CrystalSize_groupBox.setFont(QFont(self.font,self.fontsize))
        group_opts = self.GroupBox_StyleSheet(self.red, self.red_light)
        self.CrystalSize_groupBox.setStyleSheet(group_opts)
        
        self.CrystalSize_button = QPushButton('D (\u212b)')
        self.CrystalSize_button.clicked.connect(self.reset_D)
        button_opts = self.PushButton_StyleSheet(self.red)
        self.CrystalSize_button.setStyleSheet(button_opts)
        self.CrystalSize_button.setToolTip(self.l.what['size_0'] + ': ' + self.l.what['tt_b_pars'])

        self.CrystalSize_slider = DoubleSlider(1,Qt.Horizontal)
        slider_opts = self.Slider_StyleSheet()
        self.CrystalSize_slider.setStyleSheet(slider_opts)
        self.CrystalSize_slider.setMinimum(self.D_min)
        self.CrystalSize_slider.setMaximum(self.D_max)
        self.CrystalSize_slider.setValue(self.init_D)
        self.CrystalSize_slider.setTickPosition(QSlider.NoTicks)
        self.CrystalSize_slider.valueChanged.connect(self.D_slider_change)
        self.CrystalSize_slider.setToolTip(self.l.what['tt_s_pars'])
        
        self.CrystalSize_entry = QLineEdit()
        entry_opts = self.LineEdit_StyleSheet(self.red)
        self.CrystalSize_entry.setFont(QFont(self.font,self.fontsize))
        self.CrystalSize_entry.setStyleSheet(entry_opts)
        self.CrystalSize_entry.setText('{: 6.1f}'.format(self.init_D))
        self.CrystalSize_entry.setMaxLength(6)
        self.CrystalSize_entry.setFixedWidth(60)
        self.CrystalSize_entry.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.CrystalSize_entry.editingFinished.connect(self.Size_update_slider)
        self.CrystalSize_entry.setToolTip(self.l.what['tt_e_pars'])
        
        CrystalSize_layout = QHBoxLayout()
        CrystalSize_layout.addSpacing(5)
        CrystalSize_layout.addWidget(self.CrystalSize_button)
        CrystalSize_layout.addWidget(self.CrystalSize_slider)
        CrystalSize_layout.addWidget(self.CrystalSize_entry)

        self.CrystalSize_groupBox.setLayout(CrystalSize_layout)

        return self.CrystalSize_groupBox

    def reset_D(self):
        """ function called when clicked the crystal size button, in order to return to initial value"""
        self.D = self.init_D
        self.CrystalSize_slider.setValue(self.D)

    def D_slider_change(self):
        """ function called when the crystal size slider is changed"""
        self.D = self.CrystalSize_slider.value()
        self.CrystalSize_entry.setText('{: 6.1f}'.format(self.D))
        self.update(ul=False, xpd = True, en = False)
        
    def Size_update_slider(self):
        """ function called when the edit box text of the crystal size is changed"""
        val = self.CrystalSize_entry.text()
        try:
            self.D = float(val)
            self.CrystalSize_slider.setValue(float(self.CrystalSize_entry.text()))
            self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, self.l.what['problem_a'], self.l.what['size_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.CrystalSize_entry.setText('{: 6.1f}'.format(self.D))

    def include_InfoFrame(self):
        """ function that includes the information frame. In future updates, it will be removed"""
        self.Info_groupBox = QGroupBox(self.l.what['info'], self) 
        self.Info_groupBox.setFont(QFont(self.font,self.fontsize))
        group_opts = self.GroupBox_StyleSheet(self.red, self.red_light)
        self.Info_groupBox.setStyleSheet(group_opts)
        
        self.Info_label = QLabel()
        self.Info_label.setFont(QFont(self.font,self.fontsize))
        self.Info_label.setText(self.l.what['welcome'])
        self.Info_label.setMinimumWidth(150)
        label_opts = 'QLabel { background: #d4d4d4; border-radius: 3px;}'
        self.Info_label.setStyleSheet(label_opts)
        self.Info_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        
        Info_layout = QVBoxLayout()
        Info_layout.addSpacing(5)
        Info_layout.addWidget(self.Info_label)
        
        self.Info_groupBox.setLayout(Info_layout)
        
        return self.Info_groupBox

    def Update_Info_label(self, text = '', bkg = '#d4d4d4'):
        """function that updates the text and background of the information frame """
        label_opts = r'QLabel { background: ' + '{}'.format(bkg) + r'; border-radius: 3px;}'
        self.Info_label.setStyleSheet(label_opts)
        self.Info_label.setText(text)

    def include_Params_opts(self):
        """ function that includes the groupbox containing the button of the parameter's settings """
        self.Params_opts_groupBox = QGroupBox('', self) 
        group_opts = self.GroupBox_StyleSheet(self.red, self.red_light)
        self.Params_opts_groupBox.setStyleSheet(group_opts)
        
        self.settings_params_button = QPushButton('')
        self.settings_params_button.clicked.connect(self.settings_params_opts)
        button_opts = self.PushButton_Toolbox_StyleSheet(self.sl_col_1)
        self.settings_params_button.setStyleSheet(button_opts)
        self.settings_params_button.setToolTip(self.l.what['params_settings'])
        icon = qta.icon('ri.settings-5-fill', color='k', scale_factor = 1.3)
        self.settings_params_button.setIcon(icon)
        
        Params_opts_layout = QVBoxLayout()
        Params_opts_layout.addWidget(self.settings_params_button)
        
        self.Params_opts_groupBox.setLayout(Params_opts_layout)
        
        return self.Params_opts_groupBox

    def settings_params_opts(self):
        """ function triggered when the setting button is clicked. It opens a window with some options """
        self.params_options = { 'E_ini':self.init_E, 
                                'E_min':self.E_min, 
                                'E_max':self.E_max,
                                
                                'CrysSize_ini': self.init_D,
                                'CrysSize_min': self.D_min,
                                'CrysSize_max': self.D_max,

                                'a_ini':self.par0['a'], 
                                'a_min':self.par_min['a'], 
                                'a_max':self.par_max['a'], 
                                
                                'b_ini':self.par0['b'], 
                                'b_min':self.par_min['b'], 
                                'b_max':self.par_max['b'], 

                                'c_ini':self.par0['c'],
                                'c_min':self.par_min['c'],
                                'c_max':self.par_max['c'],

                                'alpha_ini': self.par0['alpha'],
                                'alpha_min': self.par_min['alpha'],
                                'alpha_max': self.par_max['alpha'],

                                'beta_ini': self.par0['beta'],
                                'beta_min': self.par_min['beta'],
                                'beta_max': self.par_max['beta'],

                                'gamma_ini': self.par0['gamma'],
                                'gamma_min': self.par_min['gamma'],
                                'gamma_max': self.par_max['gamma']                     
                                }
        self.params_functions = { 'E_ini':self.set_init_E, 
                                'E_min':self.set_E_min, 
                                'E_max':self.set_E_max,
                                
                                'CrysSize_ini': self.set_init_D,
                                'CrysSize_min': self.set_D_min,
                                'CrysSize_max': self.set_D_max,

                                'a_ini':self.set_init_a, 
                                'b_ini':self.set_init_b, 
                                'c_ini':self.set_init_c,
                                'alpha_ini': self.set_init_alpha,
                                'beta_ini': self.set_init_beta,
                                'gamma_ini': self.set_init_gamma,

                                'a_min':self.set_a_min, 
                                'b_min':self.set_b_min, 
                                'c_min':self.set_c_min,
                                'alpha_min': self.set_alpha_min,
                                'beta_min': self.set_beta_min,
                                'gamma_min': self.set_gamma_min,
                                
                                'a_max':self.set_a_max, 
                                'b_max':self.set_b_max, 
                                'c_max':self.set_c_max,
                                'alpha_max': self.set_alpha_max,
                                'beta_max': self.set_beta_max,
                                'gamma_max': self.set_gamma_max
                                }
        self.params_names = {   'E_ini':'E (keV) ini', 
                                'E_min':'E (keV) min', 
                                'E_max':'E (keV) max',
                                
                                'CrysSize_ini':'D (\u212b) ini',
                                'CrysSize_min':'D (\u212b) min',
                                'CrysSize_max':'D (\u212b) max',

                                'a_ini':'a (\u212b) ini', 
                                'b_ini':'b (\u212b) ini', 
                                'c_ini':'c (\u212b) ini',
                                'alpha_ini':'\u03b1 (\u00b0) ini',
                                'beta_ini':'\u03b2 (\u00b0) ini',
                                'gamma_ini':'\u03b3 (\u00b0) ini',

                                'a_min':'a (\u212b) min', 
                                'b_min':'b (\u212b) min', 
                                'c_min':'c (\u212b) min',
                                'alpha_min':'\u03b1 (\u00b0) min',
                                'beta_min':'\u03b2 (\u00b0) min',
                                'gamma_min':'\u03b3 (\u00b0) min',
                                
                                'a_max':'a (\u212b) max', 
                                'b_max':'b (\u212b) max', 
                                'c_max':'c (\u212b) max',
                                'alpha_max':'\u03b1 (\u00b0) max',
                                'beta_max':'\u03b2 (\u00b0) max',
                                'gamma_max':'\u03b3 (\u00b0) max'
                                }
        self.params_tooltips = {'E_ini':self.l.what['param_energy_ini'], 
                                'E_min':self.l.what['param_energy_minmax'].format(minmax = self.l.what['min']),
                                'E_max':self.l.what['param_energy_minmax'].format(minmax = self.l.what['max']),
                                
                                'CrysSize_ini':self.l.what['param_CrysSize_ini'],
                                'CrysSize_min':self.l.what['param_CrysSize_minmax'].format(minmax = self.l.what['min']),
                                'CrysSize_max':self.l.what['param_CrysSize_minmax'].format(minmax = self.l.what['max']),

                                'a_ini':self.l.what['lattice_param_ini'],
                                'b_ini':self.l.what['lattice_param_ini'],
                                'c_ini':self.l.what['lattice_param_ini'],
                                'alpha_ini':self.l.what['lattice_param_ini'],
                                'beta_ini':self.l.what['lattice_param_ini'],
                                'gamma_ini':self.l.what['lattice_param_ini'],

                                'a_min':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['min']),
                                'b_min':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['min']),
                                'c_min':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['min']),
                                'alpha_min':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['min']),
                                'beta_min':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['min']),
                                'gamma_min':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['min']),
                                
                                'a_max':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['max']),
                                'b_max':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['max']),
                                'c_max':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['max']),
                                'alpha_max':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['max']),
                                'beta_max':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['max']),
                                'gamma_max':self.l.what['lattice_param_minmax'].format(minmax = self.l.what['max'])
                                }
        self.params_types = {   'E_ini':'float',
                                'E_min':'float',
                                'E_max':'float',
                                
                                'CrysSize_ini':'float',
                                'CrysSize_min':'float',
                                'CrysSize_max':'float',

                                'a_ini':'float',
                                'b_ini':'float',
                                'c_ini':'float',
                                'alpha_ini':'float',
                                'beta_ini':'float',
                                'gamma_ini':'float',

                                'a_min':'float',
                                'b_min':'float',
                                'c_min':'float',
                                'alpha_min':'float',
                                'beta_min':'float',
                                'gamma_min':'float',
                                
                                'a_max':'float',
                                'b_max':'float',
                                'c_max':'float',
                                'alpha_max':'float',
                                'beta_max':'float',
                                'gamma_max':'float'
                                }
        params_min_lims = {     'E_ini':self.E_min,
                                'E_min':2,
                                'E_max':self.E_min,
                                
                                'CrysSize_ini':self.D_min,
                                'CrysSize_min':10,
                                'CrysSize_max':self.D_min,

                                'a_ini':self.par_min['a'],
                                'b_ini':self.par_min['b'],
                                'c_ini':self.par_min['c'],
                                'alpha_ini':self.par_min['alpha'],
                                'beta_ini':self.par_min['beta'],
                                'gamma_ini':self.par_min['gamma'],

                                'a_min':1,
                                'b_min':1,
                                'c_min':1,
                                'alpha_min':30,
                                'beta_min':30,
                                'gamma_min':30,
                                
                                'a_max':self.par_min['a'],
                                'b_max':self.par_min['b'],
                                'c_max':self.par_min['c'],
                                'alpha_max':self.par_min['alpha'],
                                'beta_max':self.par_min['beta'],
                                'gamma_max':self.par_min['gamma']
                                }
        params_max_lims = {     'E_ini':self.E_max,
                                'E_min':self.E_max,
                                'E_max':40,
                                
                                'CrysSize_ini':self.D_max,
                                'CrysSize_min':self.D_max,
                                'CrysSize_max':10000,

                                'a_ini':self.par_max['a'],
                                'b_ini':self.par_max['b'],
                                'c_ini':self.par_max['c'],
                                'alpha_ini':self.par_max['alpha'],
                                'beta_ini':self.par_max['beta'],
                                'gamma_ini':self.par_max['gamma'],

                                'a_min':self.par_max['a'],
                                'b_min':self.par_max['b'],
                                'c_min':self.par_max['c'],
                                'alpha_min':self.par_max['alpha'],
                                'beta_min':self.par_max['beta'],
                                'gamma_min':self.par_max['gamma'],
                                
                                'a_max':40,
                                'b_max':40,
                                'c_max':40,
                                'alpha_max':179,
                                'beta_max':179,
                                'gamma_max':179
                                }
        color =    {            'E_ini':self.red_red,
                                'E_min':self.red,
                                'E_max':self.red,
                                
                                'CrysSize_ini':self.red_red,
                                'CrysSize_min':self.red,
                                'CrysSize_max':self.red,

                                'a_ini':self.green_dark,
                                'b_ini':self.green_dark,
                                'c_ini':self.green_dark,
                                'alpha_ini':self.green_dark,
                                'beta_ini':self.green_dark,
                                'gamma_ini':self.green_dark,

                                'a_min':self.green,
                                'b_min':self.green,
                                'c_min':self.green,
                                'alpha_min':self.green,
                                'beta_min':self.green,
                                'gamma_min':self.green,
                                
                                'a_max':self.green,
                                'b_max':self.green,
                                'c_max':self.green,
                                'alpha_max':self.green,
                                'beta_max':self.green,
                                'gamma_max':self.green
                                }
        save =  {               'E_ini':True,
                                'E_min':True,
                                'E_max':True,
                                
                                'CrysSize_ini':True,
                                'CrysSize_min':True,
                                'CrysSize_max':True,

                                'a_ini':False,
                                'b_ini':False,
                                'c_ini':False,
                                'alpha_ini':False,
                                'beta_ini':False,
                                'gamma_ini':False,

                                'a_min':False,
                                'b_min':False,
                                'c_min':False,
                                'alpha_min':False,
                                'beta_min':False,
                                'gamma_min':False,
                                
                                'a_max':False,
                                'b_max':False,
                                'c_max':False,
                                'alpha_max':False,
                                'beta_max':False,
                                'gamma_max':False,
                                }
        title = "Parameter Options"
        self.w1 = PopUpOpt(self, self.params_options, self.params_functions, self.params_names, self.params_tooltips, self.params_types, params_min_lims, params_max_lims, color, title, save)
        self.w1.setGeometry(QRect(100, 100, 200, 300))
        self.w1.show()

    def set_init_E(self, init_E):
        """ Function that defines the default value for the energy """
        self.init_E = init_E
        return self.init_E

    def set_E_min(self, E_min):
        """ function that defines the minimum value for the energy, used in the slider"""
        self.E_min = E_min
        self.E_slider.setMinimum(self.E_min)
        return self.E_min

    def set_E_max(self, E_max):
        """ function that defines the maximum value for the energy, used in the slider"""
        self.E_max = E_max
        self.E_slider.setMaximum(self.E_max)
        return self.E_max
        
    def set_init_D(self, init_D):
        """ Function that defines the default value for the crystal size """
        self.init_D = init_D
        return self.init_D

    def set_D_min(self, D_min):
        """ function that defines the minimum value for the crystal size, used in the slider"""
        self.D_min = D_min
        self.CrystalSize_slider.setMinimum(self.D_min)
        return self.D_min

    def set_D_max(self, D_max):
        """ function that defines the maximum value for the crystal size, used in the slider"""
        self.D_max = D_max
        self.CrystalSize_slider.setMaximum(self.D_max)
        return self.D_max

    def set_init_a(self, init_a):
        """ function that defines the default value for the lattice parameter 'a' """
        self.par0['a'] = init_a
        return self.par0['a']

    def set_a_min(self, a_min):
        """ function that defines the minimum value for the lattice parameter 'a', used in the slider"""
        self.par_min['a'] = a_min
        self. LatticeParams_slider['a'].setMinimum(self.par_min['a'])
        return self.par_min['a']

    def set_a_max(self, a_max):
        """ function that defines the maximum value for the lattice parameter 'a', used in the slider"""
        self.par_max['a'] = a_max
        self. LatticeParams_slider['a'].setMaximum(self.par_max['a'])
        return self.par_max['a']

    def set_init_b(self, init_b):
        """ function that defines the default value for the lattice parameter 'b' """
        self.par0['b'] = init_b
        return self.par0['b']

    def set_b_min(self, b_min):
        """ function that defines the minimum value for the lattice parameter 'b', used in the slider"""
        self.par_min['b'] = b_min
        self. LatticeParams_slider['b'].setMinimum(self.par_min['b'])
        return self.par_min['b']

    def set_b_max(self, b_max):
        """ function that defines the maximum value for the lattice parameter 'b', used in the slider"""
        self.par_max['b'] = b_max
        self. LatticeParams_slider['b'].setMaximum(self.par_max['b'])
        return self.par_max['b']

    def set_init_c(self, init_c):
        """ function that defines the default value for the lattice parameter 'c' """
        self.par0['c'] = init_c
        return self.par0['c']

    def set_c_min(self, c_min):
        """ function that defines the minimum value for the lattice parameter 'c', used in the slider"""
        self.par_min['c'] = c_min
        self. LatticeParams_slider['c'].setMinimum(self.par_min['c'])
        return self.par_min['c']

    def set_c_max(self, c_max):
        """ function that defines the maximum value for the lattice parameter 'c', used in the slider"""
        self.par_max['c'] = c_max
        self. LatticeParams_slider['c'].setMaximum(self.par_max['c'])
        return self.par_max['c']

    def set_init_alpha(self, init_alpha):
        """ function that defines the default value for the angle 'alpha' between lattices 'b' and 'c' """
        self.par0['alpha'] = init_alpha
        return self.par0['alpha']

    def set_alpha_min(self, alpha_min):
        """ function that defines the minimum value for the angle 'alpha', used in the slider"""
        self.par_min['alpha'] = alpha_min
        self. LatticeParams_slider['alpha'].setMinimum(self.par_min['alpha'])
        return self.par_min['alpha']

    def set_alpha_max(self, alpha_max):
        """ function that defines the maximum value for the angle 'alpha', used in the slider"""
        self.par_max['alpha'] = alpha_max
        self. LatticeParams_slider['alpha'].setMaximum(self.par_max['alpha'])
        return self.par_max['alpha']

    def set_init_beta(self, init_beta):
        """ function that defines the default value for the angle 'beta' between lattices 'a' and 'c' """
        self.par0['beta'] = init_beta
        return self.par0['beta']

    def set_beta_min(self, beta_min):
        """ function that defines the minimum value for the angle 'beta', used in the slider"""
        self.par_min['beta'] = beta_min
        self. LatticeParams_slider['beta'].setMinimum(self.par_min['beta'])
        return self.par_min['beta']

    def set_beta_max(self, beta_max):
        """ function that defines the maximum value for the angle 'beta', used in the slider"""
        self.par_max['beta'] = beta_max
        self. LatticeParams_slider['beta'].setMaximum(self.par_max['beta'])
        return self.par_max['beta']

    def set_init_gamma(self, init_gamma):
        """ function that defines the default value for the angle 'gamma' between lattice 'a' and 'b' """
        self.par0['gamma'] = init_gamma
        return self.par0['gamma']

    def set_gamma_min(self, gamma_min):
        """ function that defines the minimum value for the angle 'gamma', used in the slider"""
        self.par_min['gamma'] = gamma_min
        self. LatticeParams_slider['gamma'].setMinimum(self.par_min['gamma'])
        return self.par_min['gamma']

    def set_gamma_max(self, gamma_max):
        """ function that defines the maximum value for the angle 'gamma', used in the slider"""
        self.par_max['gamma'] = gamma_max
        self. LatticeParams_slider['gamma'].setMaximum(self.par_max['gamma'])
        return self.par_max['gamma']



    # crystal figure functions and options
    def include_CrystalFigure(self):
        """ function that includes the groupbox of the crystal figure, with all components"""
        # just the name of the group
        self.CrystalFigure_groupBox = QGroupBox(self.l.what['crystal'], self) 
        self.CrystalFigure_groupBox.setFont(QFont(self.font,self.fontsize))
        group_opts = self.GroupBox_StyleSheet(self.gray, self.gray_light)
        self.CrystalFigure_groupBox.setStyleSheet(group_opts)

        self.CrystalFigure = plt.figure()
        self.Crystalax = plt.axes([0.08, 0.08, 0.9, 0.9],projection='3d')
        self.Crystalcanvas = FigureCanvas(self.CrystalFigure) 

        self.Vis_slider = DoubleSlider(1,Qt.Vertical)
        slider_opts = self.Slider_StyleSheet('v')
        self.Vis_slider.setStyleSheet(slider_opts)
        self.Vis_slider.setMinimum(self.plotlimits_min)
        self.Vis_slider.setMaximum(self.plotlimits_max)
        self.Vis_slider.setValue(self.plotlimits)
        self.Vis_slider.setTickPosition(QSlider.NoTicks)
        self.Vis_slider.valueChanged.connect(self.change_limits)
        self.Vis_slider.setToolTip(self.l.what['tt_s_uc'])

        self.Edge_check = QCheckBox('')
        #self.Edge_check.setFont(QFont(font,fontsize, 5))
        check_opts = self.Checkbox_StyleSheet(color = '#ff0000')
        self.Edge_check.setStyleSheet(check_opts)
        self.Edge_check.setChecked(True)
        self.Edge_check.stateChanged.connect(self.check_edge_TF)
        self.Edge_check.setToolTip(self.l.what['crysedge'])
        
        self.Face_check = QCheckBox('')
        check_opts = self.Checkbox_StyleSheet(color = '#00ffff')
        self.Face_check.setStyleSheet(check_opts)
        self.Face_check.setChecked(True)
        self.Face_check.stateChanged.connect(self.check_face_TF)
        self.Face_check.setToolTip(self.l.what['crysface'])
        
        
        self.Atoms_check = QCheckBox('')
        check_opts = self.Checkbox_StyleSheet(color = self.blue_dark)
        self.Atoms_check.setStyleSheet(check_opts)
        self.Atoms_check.setChecked(True)
        self.Atoms_check.stateChanged.connect(self.check_showhideatoms_TF)
        self.Atoms_check.setToolTip(self.l.what['crysatoms'])

        self.Extended_cells_check = QCheckBox('')
        check_opts = self.Checkbox_StyleSheet(color = '#ffaa00')
        self.Extended_cells_check.setStyleSheet(check_opts)
        self.Extended_cells_check.setChecked(False)
        self.Extended_cells_check.stateChanged.connect(self.check_extended_cells_TF)
        self.Extended_cells_check.setToolTip(self.l.what['extended_cells'])
        
        self.Include_edge_atoms_check = QCheckBox('')
        check_opts = self.Checkbox_StyleSheet(color = '#aaaa00')
        self.Include_edge_atoms_check.setStyleSheet(check_opts)
        self.Include_edge_atoms_check.setChecked(False)
        self.Include_edge_atoms_check.stateChanged.connect(self.check_add_edge_atoms_TF)
        self.Include_edge_atoms_check.setToolTip(self.l.what['include_edge_atms'])
        
        
        
        
        # Setting the layout:
        Figure_layout = QGridLayout()
        Figure_layout.addWidget(self.Crystalcanvas, 0, 1, 8, 10)
        Figure_layout.addWidget(self.Vis_slider, 0, 0, 4, 1)
        Figure_layout.addWidget(self.Atoms_check, 4, 0, 1, 1)
        Figure_layout.addWidget(self.Edge_check, 5, 0, 1, 1)
        Figure_layout.addWidget(self.Face_check, 6, 0, 1, 1)
        Figure_layout.addWidget(self.Extended_cells_check, 7, 0, 1, 1)
        #Figure_layout.addWidget(self.Include_edge_atoms_check, 8, 0, 1, 1)
        self.CrystalFigure_groupBox.setLayout(Figure_layout)
        
        return self.CrystalFigure_groupBox

    def change_limits(self):
        """Function that changes the limits of the 3D plot, based ont he slider value"""
        self.plotlimits = self.Vis_slider.value()
        self.update(ul=False, xpd = False, en = False)
        
    def check_edge_TF(self):
        """ calls the update function with no calculations just changing the crystal visualization removing or including the edge of the unit cell """
        self.update(ul=False, xpd = False, en = False)

    def check_face_TF(self):
        """ calls the update function with no calculations just changing the crystal visualization removing or including the face of the unit cell """
        self.update(ul=False, xpd = False, en = False)

    def check_showhideatoms_TF(self):
        """ calls the update function with no calculations just changing the crystal visualization removing or including the atoms of the unit cell """
        self.update(ul=False, xpd = False, en = False)

    def check_extended_cells_TF(self):
        """ calls the update function with no calculations just changing the crystal visualization removing or including the atoms of extra unit cells  """
        self.update(ul=False, xpd = False, en = False)
        
    def check_add_edge_atoms_TF(self):
        """ calls the update function with no calculations just changing the crystal visualization removing or including the atoms from the edge of the unit cell. Not sure that I have imp0lemented that!! :-o """
        self.update(ul=False, xpd = False, en = False)
        
    def calc_HKL_planes(self):
        """ function that calculates the equations of the HKL planes, to include them or not in the visualization of the unit cell"""
        h = int(self.le_H.text())
        k = int(self.le_K.text())
        l = int(self.le_L.text())
        #lim = self.plotlimits
        #lim2 = lim/5.
        mesh = np.arange(-0.2,1.21, 0.2)
        x0, x1 = np.meshgrid(mesh, mesh)
        d = []
        delta = 0.1
        diag = np.array(self.set_pos(1,1,1))+delta
        #dist = np.sqrt(np.dot(diag,[h,k,l]))
        #d_HKL = 2*np.pi/self.Qhkl(h,k,l)
        _alpha = 0.75
        #while n*d_HKL <= 5*dist: n+=1
        if h != 0:
            for i in range(-4,4,1): 
                x = (-x0*k -x1*l +i)/h
                [xx, yy, zz] = self.set_pos(x,x0,x1)
                if np.all([np.any(xx <= diag[0]), np.any(xx >= -delta), np.any(yy <= diag[1]), np.any(yy >= -delta), np.any(zz <= diag[2]), np.any(zz >= -delta)]):
                    self.curv = self.Crystalax.plot_surface(xx, yy, zz, color = self.magenta_light, alpha = _alpha )
        elif k != 0:
            for i in range(-4,4,1): 
                y = (-x0*h -x1*l +i)/k
                [xx, yy, zz] = self.set_pos(x0,y,x1)
                if np.all([np.any(xx <= diag[0]), np.any(xx >= -delta), np.any(yy <= diag[1]), np.any(yy >= -delta), np.any(zz <= diag[2]), np.any(zz >= -delta)]):
                    self.curv = self.Crystalax.plot_surface(xx, yy, zz, color = self.magenta_light, alpha = _alpha )
        elif l != 0:
            for i in range(-4,4,1): 
                z = (-x0*h -x1*k +i)/l
                [xx, yy, zz] = self.set_pos(x0,x1,z)
                if np.all([np.any(xx <= diag[0]), np.any(xx >= -delta), np.any(yy <= diag[1]), np.any(yy >= -delta), np.any(zz <= diag[2]), np.any(zz >= -delta)]):
                    self.curv = self.Crystalax.plot_surface(xx, yy, zz, color = self.magenta_light, alpha = _alpha )
        self.Crystalcanvas.draw()







    def updateQs(self):
        """ Updates the lower and higher Q values, depending on the tth minimum and maximum.    """
        self.Qi = self.Q(self.tth_min)
        self.Qf = self.Q(self.tth_max)

    def Q(self,tth): 
        """ returns the Q depending on the tth value entered  """
        return 4*np.pi/self.Wvl_slider.value()*np.sin(tth*np.pi/360.)

    # base atoms functions and options
    def include_BaseAtoms(self):
        """ function that includes the groupbox of the atoms of the unit cell, with all components"""
        # just the name of the group
        self.BaseAtoms_groupBox = QGroupBox(self.l.what['base'], self)
        self.BaseAtoms_groupBox.setFont(QFont(self.font,self.fontsize))
        self.BaseAtoms_groupBox.setStyleSheet(self.GroupBox_StyleSheet(self.color_atom0, self.blue_light))

        #self.BaseAtoms_settings_button = QPushButton('')
        #self.BaseAtoms_settings_button.clicked.connect(self.BaseAtoms_settings_opts)
        #button_opts = self.PushButton_Toolbox_StyleSheet(self.sl_col_1)
        #self.BaseAtoms_settings_button.setStyleSheet(button_opts)
        #self.BaseAtoms_settings_button.setToolTip(self.l.what['baseatoms_settings'])
        #icon = qta.icon('ri.settings-5-fill', color='k', scale_factor = 1.3)
        #self.BaseAtoms_settings_button.setIcon(icon)
        
        self.BaseAtoms_entry_0 = QLineEdit(self.baseAtoms[0])
        self.BaseAtoms_entry_0.setStyleSheet(self.LineEdit_StyleSheet(self.color_atom0))
        self.BaseAtoms_entry_0.setFont(QFont(self.font,self.fontsize))
        self.BaseAtoms_entry_0.setMaxLength(2)
        self.BaseAtoms_entry_0.editingFinished.connect(lambda atom = 0: self.newAtomEntered(atom))
        self.BaseAtoms_entry_0.setFixedWidth(40);
        self.BaseAtoms_entry_0.setToolTip(self.l.what['label_0'])
        self.BaseAtoms_entry_0.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        
        self.BaseAtoms_color = QPushButton('')
        self.BaseAtoms_color.clicked.connect(self.getBaseAtom_color)
        button_opts = self.PushButton_StyleSheet(self.color_atom0)
        button_opts2 = self.PushButton_StyleSheet2(self.color_atom0)
        icon = qta.icon('msc.symbol-color', color='k', scale_factor = 1.1)
        self.BaseAtoms_color.setIcon(icon)
        self.BaseAtoms_color.setStyleSheet(button_opts2)
        self.BaseAtoms_color.setToolTip(self.l.what['color_0'])

        self.BaseAtoms_entry_0_size = QLineEdit('{}'.format(self.sizes_init[0]/100.))
        self.BaseAtoms_entry_0_size.setStyleSheet(self.LineEdit_StyleSheet(self.color_atom0))
        self.BaseAtoms_entry_0_size.setFont(QFont(self.font,self.fontsize))
        self.BaseAtoms_entry_0_size.setMaxLength(4)
        self.BaseAtoms_entry_0_size.editingFinished.connect(lambda atom = 0: self.newAtomSizeEntered(atom))
        self.BaseAtoms_entry_0_size.setFixedWidth(40);
        self.BaseAtoms_entry_0_size.setToolTip(self.l.what['atom_size'])
        self.BaseAtoms_entry_0_size.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        
        
        self.BaseAtoms_button_addAtom = QPushButton('+ {}'.format(self.l.what['atm']))
        self.BaseAtoms_button_addAtom.setFont(QFont(self.font,self.fontsize))
        self.BaseAtoms_button_addAtom.clicked.connect(self.add_Atom)
        self.BaseAtoms_button_addAtom.setStyleSheet(button_opts)
        self.BaseAtoms_button_addAtom.setToolTip(self.l.what['addAtoms'])
        self.BaseAtoms_button_remAtom = QPushButton('- {}'.format(self.l.what['atm']))
        self.BaseAtoms_button_remAtom.setFont(QFont(self.font,self.fontsize))
        self.BaseAtoms_button_remAtom.clicked.connect(self.rem_Atom)
        self.BaseAtoms_button_remAtom.setStyleSheet(button_opts)
        self.BaseAtoms_button_remAtom.setToolTip(self.l.what['remAtoms'])

        self.BaseAtoms_tabs = QTabWidget()



        # North  = 0; South  = 1; West  = 2; East = 3
        c3 = 'min-height: 15ex;padding: 2px;}'
        b = 'QTabWidget::tab-bar { left: 0px; top: 15px}'
        c2 = 'border-bottom-color: #C2C7CB; border-top-left-radius: 5px; border-bottom-left-radius: 5px;'
        
        self.BaseAtoms_tabs.setTabPosition(2)
        
        a = 'QTabWidget::pane { border-top: 2px solid #C2C7CB; border-left: 2px solid #C2C7CB; border-right: 2px solid #C2C7CB; border-bottom: 2px solid #C2C7CB;}'
        c = 'QTabBar::tab { background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #E1E1E1, '
        c1 = 'stop: 0.4 #DDDDDD, stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3); border: 2px solid #C4C4C3; '
        
        d = 'QTabBar::tab:selected, QTabBar::tab:hover { background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,'
        d1 = 'stop: 0 {}, stop:  1.0 {});'.format(self.blue,self.blue_light)+'}'
        e = 'QTabBar::tab:selected {border-color: '+ '{};border-bottom-color: {}'.format(self.blue, self.blue)+';}'
        f = 'QTabBar::tab:!selected {margin-top: 2px; }'
        tab_opts = a + b + c + c1+ c2 + c3 + d + d1 + e + f
        self.BaseAtoms_tabs.setStyleSheet(tab_opts)

        
        self.BaseAtoms_tab1 = QWidget()
        self.BaseAtoms_tab2 = QWidget()
        self.BaseAtoms_tab3 = QWidget()

        self.layout = [QVBoxLayout(self.BaseAtoms_tab1),QVBoxLayout(self.BaseAtoms_tab2),QVBoxLayout(self.BaseAtoms_tab3)]
        
        self.BaseAtoms_tabs.addTab(self.BaseAtoms_tab1,"pos 1-3")
        self.BaseAtoms_tabs.addTab(self.BaseAtoms_tab2,"pos 4-6")
        self.BaseAtoms_tabs.addTab(self.BaseAtoms_tab3,"pos 7-9")
        for i in range (3): self.BaseAtoms_tabs.setTabEnabled (i, False)
        
        BaseAtoms_layout = QGridLayout()
        BaseAtoms_layout.setContentsMargins(5, 15, 5, 5)
        
        #BaseAtoms_layout.addWidget(self.BaseAtoms_settings_button, 1,0)
        BaseAtoms_layout.addWidget(self.BaseAtoms_entry_0, 1, 0)
        BaseAtoms_layout.addWidget(self.BaseAtoms_color,1,1)
        BaseAtoms_layout.addWidget(self.BaseAtoms_entry_0_size,1,2)
        BaseAtoms_layout.addWidget(self.BaseAtoms_button_addAtom, 1, 10,1,2)
        BaseAtoms_layout.addWidget(self.BaseAtoms_button_remAtom, 1, 12,1,2)
        BaseAtoms_layout.addWidget(self.BaseAtoms_tabs,2,0,-1,-1)
        self.BaseAtoms_groupBox.setLayout(BaseAtoms_layout)

        return self.BaseAtoms_groupBox

    def newAtomEntered(self, atom, init = False):
        """ function called when a new text is entered in the text box of the atoms' names. This funtion is called in the beginning of the program; so to do not update so many times when an atom is included, I put this option 'init = False' """
        if atom == 0:
            func = self.BaseAtoms_entry_0
        else:
            func = self.AddAtoms_entry_At[atom]
        try:
            eval('xu.materials.elements.{}'.format(func.text()))
            self.Atom_types.update({atom:func.text()})
            self.last_atom = func.text()
            if not init: self.update(ul=False, xpd = True, en = True)
        except:
            text0 = func.text()
            func.setText(self.last_atom)
            text = '"{}" {} {}'.format(text0, self.l.what['problem_a'], self.l.what['atm'])
            self.Update_Info_label(text = text, bkg = self.red)

    def newAtomSizeEntered(self, atom):
        """ function called when a new text is entered in the text box of the atoms' size. I've put a limit for the sizes, but, actually, never tried what is the real value of the limit.  """
        if atom == 0:
            func = self.BaseAtoms_entry_0_size
        else:
            func = self.AddAtoms_entry_At_size[atom]
        try:
            numb = float(func.text())
            if numb < 1 or numb > 10:
                text = '"{}" {} {}'.format(numb, self.l.what['problem_a'], self.l.what['atom_size2'])
                self.Update_Info_label(text = text, bkg = self.red)
                func.setText('{}'.format(self.sizes_init[atom]/100.))
            else:
                self.sizes_init[atom] = numb*100
        except:
            text = '"{}" {} {}'.format(func.text(), self.l.what['problem_a'], self.l.what['atom_size2'])
            self.Update_Info_label(text = text, bkg = self.red)
            func.setText('{}'.format(self.sizes_init[atom]/100))
        self.update(ul=False, xpd = False, en = False)

    def add_Atom(self, init = False):
        """ function that prepares to include an atom in the unit cell. The limit of 10 atoms is here. One can change it here. I do not believe this include more atoms would be a problem; however, 10 is a very large value"""
        self.additional_atoms += 1
        if self.additional_atoms>9: 
            print ('maximum of 10 atoms')
            self.additional_atoms = 9
        else: 
            self.include_atom(init = init)
            if not init: self.rescale()

    def rem_Atom(self):
        """ function that prepares to remove an atom of the unit cell. """
        self.additional_atoms -= 1
        if self.additional_atoms < 0: self.additional_atoms = 0
        else: 
            self.exclude_atom()
            self.rescale()

    def include_atom(self, init = False):
        """ function that effectively includes an atom on the unit cell. it call ohter function to create and prepare the sliders and tabs """
        i = self.additional_atoms
        j = (i-1)//3
        k = (i-1)%3
        if init:
            color = self.color_init[i]
            pos_x = self.pos_x_init[i]
            pos_y = self.pos_y_init[i]
            pos_z = self.pos_z_init[i]
            atom_ = self.baseAtoms[i] 
        else:
            color = self.colors[i-1]
            pos_x = 0.5
            pos_y = 0.5
            pos_z = 0.5
            atom_ = self.last_atom

        if k == 0: 
            self.BaseAtoms_tabs.setTabEnabled (j, True)
            self.BaseAtoms_tabs.setCurrentIndex(j)
        self.AddAtoms_groupBox.update({i:QGroupBox('{} {}'.format(self.l.what['atm'],i))})
        self.AddAtoms_groupBox[i].setFont(QFont(self.font,self.fontsize))
        self.AddAtoms_groupBox[i].setStyleSheet(self.GroupBox_StyleSheet(color,'#dddddd'))
        #self.AddAtoms_groupBox[i].show()
        self.AddAtoms_groupBox[i].setMaximumHeight(int(self.layout[j].geometry().getRect()[3]/3))
        
        

        
        self.AddAtoms_groupBox_layout.update({i:QGridLayout(self.AddAtoms_groupBox[i])})
        self.AddAtoms_label_x.update({i:QLabel()})
        self.AddAtoms_label_x[i].setFont(QFont(self.font,self.fontsize))
        self.AddAtoms_label_x[i].setText('{}'.format('x'))
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_label_x[i],0,1)

        self.AddAtoms_label_y.update({i:QLabel()})
        self.AddAtoms_label_y[i].setFont(QFont(self.font,self.fontsize))
        self.AddAtoms_label_y[i].setText('{}'.format('y'))
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_label_y[i],1,1)
        
        self.AddAtoms_label_z.update({i:QLabel()})
        self.AddAtoms_label_z[i].setFont(QFont(self.font,self.fontsize))
        self.AddAtoms_label_z[i].setText('{}'.format('z'))
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_label_z[i],2,1)


        self.AddAtoms_slider_x.update({i:DoubleSlider(3,Qt.Horizontal)})
        slider_opts = self.Slider_StyleSheet()
        self.AddAtoms_slider_x[i].setStyleSheet(slider_opts)
        self.AddAtoms_slider_x[i].setMinimum(0)
        self.AddAtoms_slider_x[i].setMaximum(1)
        self.AddAtoms_slider_x[i].setValue(pos_x)
        self.AddAtoms_slider_x[i].setTickPosition(QSlider.NoTicks)
        self.AddAtoms_slider_x[i].valueChanged.connect(lambda value, atom = i: self.pos_x_slider_change(value, atom))
        self.AddAtoms_slider_x[i].setToolTip(self.l.what['atm'] + '{}, '.format(i) + self.l.what['posx'] + ': '.format(i) + self.l.what['tt_s_pars'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_slider_x[i],0,2,1,7)


        self.AddAtoms_entry_pos_x.update({i:QLineEdit('{}'.format(pos_x))})
        entry_opts = self.LineEdit_StyleSheet(color)
        self.AddAtoms_entry_pos_x[i].setStyleSheet(entry_opts)
        self.AddAtoms_entry_pos_x[i].setFont(QFont(self.font,self.fontsize))
        self.AddAtoms_entry_pos_x[i].setMaxLength(5)
        self.AddAtoms_entry_pos_x[i].editingFinished.connect(lambda atom = i: self.new_pos_x(atom))
        self.AddAtoms_entry_pos_x[i].setFixedWidth(60)
        self.AddAtoms_entry_pos_x[i].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.AddAtoms_entry_pos_x[i].setToolTip(self.l.what['atm'] + '{}, '.format(i) + self.l.what['posx'] + ': '.format(i) + self.l.what['tt_e_pars'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_entry_pos_x[i],0,11)
        
        
        self.AddAtoms_slider_y.update({i:DoubleSlider(3,Qt.Horizontal)})
        self.AddAtoms_slider_y[i].setStyleSheet(slider_opts)
        self.AddAtoms_slider_y[i].setMinimum(0)
        self.AddAtoms_slider_y[i].setMaximum(1)
        self.AddAtoms_slider_y[i].setValue(pos_y)
        self.AddAtoms_slider_y[i].setTickPosition(QSlider.NoTicks)
        self.AddAtoms_slider_y[i].valueChanged.connect(lambda value, atom = i: self.pos_y_slider_change(value, atom))
        self.AddAtoms_slider_y[i].setToolTip(self.l.what['atm'] + '{}, '.format(i) + self.l.what['posy'] + ': '.format(i) + self.l.what['tt_s_pars'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_slider_y[i],1,2,1,7)
        
        
        self.AddAtoms_entry_pos_y.update({i:QLineEdit('{}'.format(pos_y))})
        self.AddAtoms_entry_pos_y[i].setStyleSheet(entry_opts)
        self.AddAtoms_entry_pos_y[i].setFont(QFont(self.font,self.fontsize))
        self.AddAtoms_entry_pos_y[i].setMaxLength(5)
        self.AddAtoms_entry_pos_y[i].editingFinished.connect(lambda atom = i: self.new_pos_y(atom))
        self.AddAtoms_entry_pos_y[i].setFixedWidth(60)
        self.AddAtoms_entry_pos_y[i].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.AddAtoms_entry_pos_y[i].setToolTip(self.l.what['atm'] + '{}, '.format(i) + self.l.what['posy'] + ': '.format(i) + self.l.what['tt_e_pars'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_entry_pos_y[i],1,11)
        
        
        self.AddAtoms_slider_z.update({i:DoubleSlider(3,Qt.Horizontal)})
        self.AddAtoms_slider_z[i].setStyleSheet(slider_opts)
        self.AddAtoms_slider_z[i].setMinimum(0)
        self.AddAtoms_slider_z[i].setMaximum(1)
        self.AddAtoms_slider_z[i].setValue(pos_z)
        self.AddAtoms_slider_z[i].setTickPosition(QSlider.NoTicks)
        self.AddAtoms_slider_z[i].valueChanged.connect(lambda value, atom = i: self.pos_z_slider_change(value, atom))
        self.AddAtoms_slider_z[i].setToolTip(self.l.what['atm'] + '{}, '.format(i) + self.l.what['posz'] + ': '.format(i) + self.l.what['tt_s_pars'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_slider_z[i],2,2,1,7)
        
        
        self.AddAtoms_entry_pos_z.update({i:QLineEdit('{}'.format(pos_z))})
        self.AddAtoms_entry_pos_z[i].setStyleSheet(entry_opts)
        self.AddAtoms_entry_pos_z[i].setFont(QFont(self.font,self.fontsize))
        self.AddAtoms_entry_pos_z[i].setMaxLength(5)
        self.AddAtoms_entry_pos_z[i].editingFinished.connect(lambda atom = i: self.new_pos_z(atom))
        self.AddAtoms_entry_pos_z[i].setFixedWidth(60)
        self.AddAtoms_entry_pos_z[i].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.AddAtoms_entry_pos_z[i].setToolTip(self.l.what['atm'] + '{}, '.format(i) + self.l.what['posz'] + ': '.format(i) + self.l.what['tt_e_pars'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_entry_pos_z[i],2,11)
        
        
        self.AddAtoms_entry_At.update({i:QLineEdit(atom_)})
        self.AddAtoms_entry_At[i].setStyleSheet(entry_opts)
        self.AddAtoms_entry_At[i].setFont(QFont(self.font,self.fontsize))
        self.AddAtoms_entry_At[i].setMaxLength(2)
        self.AddAtoms_entry_At[i].editingFinished.connect(lambda atom = i: self.newAtomEntered(atom))
        self.AddAtoms_entry_At[i].setFixedWidth(40)
        self.AddAtoms_entry_At[i].setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.AddAtoms_entry_At[i].setToolTip(self.l.what['atomtype'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_entry_At[i],0,0)
        
        self.Atom_types.update({i:atom_})
        
        self.AddAtoms_color.update({i:QPushButton('')})
        self.AddAtoms_color[i].clicked.connect(lambda tf, atom = i: self.getAddAtoms_color(tf, atom))
        button_opts = self.PushButton_StyleSheet2(color)
        self.AddAtoms_color[i].setStyleSheet(button_opts)
        self.AddAtoms_color[i].setToolTip(self.l.what['color_0'])
        icon = qta.icon('msc.symbol-color', color='k', scale_factor = 1.1)
        self.AddAtoms_color[i].setIcon(icon)
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_color[i],1,0)
        
        
        if len(self.sizes_init) <= i: self.sizes_init.append(self.sizes_init[-1])
        self.AddAtoms_entry_At_size.update({i:QLineEdit('{}'.format(self.sizes_init[i]/100.))})
        self.AddAtoms_entry_At_size[i].setStyleSheet(entry_opts)
        self.AddAtoms_entry_At_size[i].setFont(QFont(self.font,self.fontsize))
        self.AddAtoms_entry_At_size[i].setMaxLength(4)
        self.AddAtoms_entry_At_size[i].editingFinished.connect(lambda atom = i: self.newAtomSizeEntered(atom))
        self.AddAtoms_entry_At_size[i].setFixedWidth(40)
        self.AddAtoms_entry_At_size[i].setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.AddAtoms_entry_At_size[i].setToolTip(self.l.what['atom_size'])
        self.AddAtoms_groupBox_layout[i].addWidget(self.AddAtoms_entry_At_size[i],2,0)
        
        self.layout[j].addWidget(self.AddAtoms_groupBox[i])
        self.layout[j].activate() #seriously: this is the only function that solved my problem. Comment it and you will see what my problem was. If you know why, send me an e-mail to explain me, please!!
        
        if not init: self.update()

    def pos_x_slider_change(self, value, atom):
        """ function called when the x position slider of an atom changes """
        self.AddAtoms_entry_pos_x[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_x[atom].value()))
        self.update(ul=False, xpd = True, en = False)

    def pos_y_slider_change(self, value, atom):
        """ function called when the y position slider of an atom changes """
        self.AddAtoms_entry_pos_y[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_y[atom].value()))
        self.update(ul=False, xpd = True, en = False)

    def pos_z_slider_change(self, value, atom):
        """ function called when the z position slider of an atom changes """
        self.AddAtoms_entry_pos_z[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_z[atom].value()))
        self.update(ul=False, xpd = True, en = False)

    def new_pos_x(self,atom):
        """ function called when the edit box of the x position of an atom changes """
        val = self.AddAtoms_entry_pos_x[atom].text()
        try:
            self.AddAtoms_slider_x[atom].setValue(float(val))
            self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, self.l.what['problem_a'], self.l.what['atm_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.AddAtoms_entry_pos_x[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_x[atom].value()))

    def new_pos_y(self,atom):
        """ function called when the edit box of the y position of an atom changes """
        val = self.AddAtoms_entry_pos_y[atom].text()
        try:
            self.AddAtoms_slider_y[atom].setValue(float(val))
            self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, self.l.what['problem_a'], self.l.what['atm_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.AddAtoms_entry_pos_y[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_y[atom].value()))

    def new_pos_z(self,atom):
        """ function called when the edit box of the z position of an atom changes """
        val = self.AddAtoms_entry_pos_z[atom].text()
        try:
            self.AddAtoms_slider_z[atom].setValue(float(val))
            self.update(ul=False, xpd = True, en = False)
        except:
            text = '"{}" {} {}'.format(val, self.l.what['problem_a'], self.l.what['atm_0'])
            self.Update_Info_label(text = text, bkg = self.red)
            self.AddAtoms_entry_pos_z[atom].setText('{: 5.3f}'.format(self.AddAtoms_slider_z[atom].value()))

    def exclude_atom(self):
        """ function preparing to exclude an atom from the unit cell """
        i = self.additional_atoms
        j = (i)//3
        k = (i)%3
        self.layout[j].itemAt(k).widget().deleteLater()
        self.update(ul=False, xpd = True, en = False)

    def getBaseAtom_color(self):
        """ function called when the color button of the first atom of the unit cell is pressed. """
        color = QColorDialog.getColor().name()
        self.BaseAtoms_entry_0.setStyleSheet(self.LineEdit_StyleSheet(color))
        self.BaseAtoms_color.setStyleSheet(self.PushButton_StyleSheet2(color))
        self.BaseAtoms_entry_0_size.setStyleSheet(self.LineEdit_StyleSheet(color))
        self.color_atom0 = color
        self.update(xpd = False, en = False)
        
    def getAddAtoms_color(self, tf, atom):
        """ function called when the color button of the atoms of the unit cell are pressed. It calls the stylesheets of the other buttons to change to new color"""
        color = QColorDialog.getColor().name()
        self.AddAtoms_color[atom].setStyleSheet(self.PushButton_StyleSheet2(color))
        self.AddAtoms_entry_At[atom].setStyleSheet(self.LineEdit_StyleSheet(color))
        self.colors[atom-1] = color
        self.AddAtoms_groupBox[atom].setStyleSheet(self.GroupBox_StyleSheet(color,'#dddddd'))
        self.AddAtoms_entry_pos_x[atom].setStyleSheet(self.LineEdit_StyleSheet(color))
        self.AddAtoms_entry_pos_y[atom].setStyleSheet(self.LineEdit_StyleSheet(color))
        self.AddAtoms_entry_pos_z[atom].setStyleSheet(self.LineEdit_StyleSheet(color))
        self.AddAtoms_entry_At_size[atom].setStyleSheet(self.LineEdit_StyleSheet(color))
        self.update(xpd = False, en = False)

    def include_base(self):
        """ function called in the beginning of the program to load all atoms of the unit cell. """
        for i in range(len(self.baseAtoms)-1):
            self.add_Atom(init = True)
            self.newAtomEntered(i+1, init = True)





    # stylesheets:
    def GroupBox_StyleSheet(self, color_name_ini, color_name_fin):
        a = 'QGroupBox {background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 ' + '{}, stop: 1 {});'.format(self.gb_bg_ini, self.gb_bg_fin)
        b = 'border: {}px solid gray; border-radius: {}px; margin-top: 2ex;'.format(int(self.fontsize/5), int(self.fontsize/2))+'}'
        c = 'QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top center; '
        d = 'padding: -4px 2px; background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 {}, stop: 1 {});'.format(color_name_ini, color_name_fin)+'}'
        return a + b + c + d
    def Slider_StyleSheet(self, orientation = 'h'):
        size = self.fontsize
        if orientation == 'h':
            a = 'QSlider::groove {border: ' + '{}px solid '.format(int(size/10))+'{}; height: {}px; background: qlineargradient(x1:0,'.format(self.sl_col_1, int(2*size/3)+1)
            b = 'y1:0, x2:0, y2:1, stop:0 {}, stop:1 {});  margin: {}px 0;'.format(self.sl_col_2, self.sl_col_3, int(size/4))+'}'
            c = 'QSlider::handle {background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 '+'{}, stop:1 {});'.format(self.sl_stp_0, self.sl_stp_1)
            d = 'border: {}px solid {}'.format(int(size/10),self.sl_col_0)+ '; width: {}px; height: {}px; margin: -4px 0; border-radius: {}px;'.format(int(size+1), int(size*3/2), int(size/4))+'}'
            return a + b + c + d
        if orientation == 'v':
            a = 'QSlider::groove {border: '+'{}px solid '.format(int(size/10))+'{}; width: {}px; background: qlineargradient(x1:0,'.format(self.sl_col_1, int(2*size/3)+1)
            b = 'y1:0, x2:0, y2:1, stop:0 {}, stop:1 {});  margin: {}px 0;'.format(self.sl_col_2, self.sl_col_3, int(size/4))+'}'
            c = 'QSlider::handle {background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 '+'{}, stop:1 {});'.format(self.sl_stp_0, self.sl_stp_1)
            d = 'border: {}px solid {}'.format(int(size/10),self.sl_col_0)+ '; height: {}px; width: {}px; margin: 0 -4px; border-radius: {}px;'.format(int(size+1), int(size*3/2), int(size/4))+'}'
            return a + b + c + d
    def LineEdit_StyleSheet(self,bg):
        size = self.fontsize
        a = 'QLineEdit {border: '+'{}px solid black; border-radius: {}px; padding: 1px; background: {};'.format(int(size/5), int(size*2/3),bg)
        b = 'selection-background-color: {}; font: bold {}px; min-width: 25px; min-height: 17 px;'.format(bg, int(size))+'}'
        return a+b
    def LEtool_StyleSheet(self, bg):
        size = self.fontsize
        a = 'QLineEdit {border: '+'{}px solid black; border-radius: {}px; padding: 1px 1px; background: {};'.format(int(size/10)+1, int(size/3),bg)
        b = 'selection-background-color: {};'.format(bg)+' font: bold {}px; max-width: 1.4em; max-height: 1em;'.format(size)+'}'
        return a+b
    def PushButton_Toolbox_StyleSheet(self, bg):
        size = self.fontsize
        a = 'QPushButton {border: '+'{}px solid black; border-radius: {}px; padding: 2px; background: {};'.format(int(size/10)+1, int(size/3),bg)
        b = 'selection-background-color: {};'.format(bg)+ '}'
        return a+b
    def PushButton_StyleSheet(self, bg):
        size = self.fontsize
        a = 'QPushButton {background-color: '+'{}; border-style: outset; border-width: {}px;'.format(bg, int(size/5)) 
        b = 'border-radius: {}px; border-color: black; font: bold {}px; min-width: 4em; padding: 0 2px; min-height: 20px;'.format(int(size*2/3),int(size))+'}'
        return a+b
    def PushButton_StyleSheet2(self, bg):
        size = self.fontsize
        a = 'QPushButton {border: '+'{}px solid black; background-color: {};'.format(int(size/5), bg) 
        b = 'border-radius: {}px; padding: 0px; min-width: 25px; min-height: 17 px;'.format(size*2/3) +'}'
        return a+b
    def Checkbox_StyleSheet(self, color):
        size = self.fontsize
        a = 'QCheckBox::indicator {width: '+'{}px;height: {}px; background-color: #ffffff; border-color: {}'.format(int(size+6), int(size+6),color) + '; border-radius: 3px;border: 2px solid;}'
        b = 'QCheckBox::indicator::unchecked {width: '+'{}px;height: {}px; border-radius: {}px; background-color: {}'.format(int(size+6),int(size+6),int(size/3),'#ffffff')+';}'
        c = 'QCheckBox::indicator:unchecked:hover {width: '+'{}px;height: {}px; border-radius: {}px; background-color: #ffffff;'.format(int(size+6),int(size+6),int(size/3))+'}'
        d = 'QCheckBox::indicator:unchecked:pressed {width: '+'{}px;height: {}px; border-radius: {}px; background-color: #dddddd;'.format(int(size+6),int(size+6),int(size/3))+'}'
        e = 'QCheckBox::indicator::checked {width: '+'{}px;height: {}px; border-radius: {}px; background-color: {}'.format(int(size+6),int(size+6),int(size/3),color)+';}'
        f = 'QCheckBox::indicator:checked:hover {width: '+'{}px;height: {}px; border-radius: {}px; background-color: {}'.format(int(size+6),int(size+6),int(size/3),color)+';}'
        g = 'QCheckBox::indicator:checked:pressed {width: '+'{}px;height: {}px; border-radius: {}px; background-color: #dddddd;'.format(int(size+6),int(size+6),int(size/3))+'}'
        return a + b + c + d + e + f + g
    # end


    def update(self, ul=False, xpd = True, en = True, from_opts = False):
        """ this function is called every time the user interacts with the main window. To decrease the time of the calculations and avoid unnecessary calculations, there are 4 (for now, at least) options which shortcut the calculations. """
        x_add = []
        y_add = []
        z_add = []
        pos_add = []
        for i in range (self.additional_atoms):
            x_add.append(self.AddAtoms_slider_x[i+1].value())
            y_add.append(self.AddAtoms_slider_y[i+1].value())
            z_add.append(self.AddAtoms_slider_z[i+1].value())
            pos_add.append([x_add[i], y_add[i], z_add[i]])
        
        self.update_unit_cell(pos_add, self.plotlimits)
        self.update_arrow()
        if ul: self.update_limits()
        if xpd: self.calculate_xpd(en, from_opts = from_opts)
        if self.showHKL_check.isChecked(): self.calc_HKL_planes()

    def rescale(self):
        """ rescales the main axis in order to show all parts of the curves. """
        max = []
        for i in range(len(self.colored_plots)):
            if self.colored_plots[i].get_visible():
                max.append(self.colored_plots[i].get_ydata().max())
        max.append(self.intensity.max())
        a = np.array(max).max()
        self.pXRDax.set_ylim(-5, a*1.2)
        self.update_arrow()
        self.XPDcanvas.draw()

    def update_unit_cell(self, pos_add, plotlimits):
        """ ths fucntion updates the unit cell 3D axis, putting the atoms in the desired positions with the desired colors, and positioning the edges and faces. Also includes the HKL planes.
            The code is big because of the inclusion of the extra unit cells. """
        #def set_pos(x, y, z): return [x*a_su+y*b_su*cg+z*c_su*cb,y*b_su*sg+z*c_su*abg,z*c_su*np.sqrt(sb*sb-abg*abg)]
        x = []
        y = []
        z = []
        pos = []
        unit_cell = []
        mult = 3./(self.plotlimits-2)
        if self.Extended_cells_check.isChecked():
            noc = 3
        else:
            noc = 2
        for _ii in range(noc):
            for _jj in range (noc):
                for _kk in range (noc):
                    pos.append(self.set_pos(_ii, _jj, _kk))
                    if _ii < 2 and _jj < 2 and _kk < 2: unit_cell.append(self.set_pos(_ii, _jj, _kk))
        self.scatter = []
        colors = []
        sizes = []
        for i in range (len(pos)):
            x.append(pos[i][0])
            y.append(pos[i][1])
            z.append(pos[i][2])
            colors.append(self.color_atom0)
            
            sizes.append(self.sizes_init[0]*mult)
        
        verts = [[pos[0],pos[1],pos[3],pos[2]], [pos[4],pos[5],pos[7],pos[6]], 
                [pos[0],pos[1],pos[5],pos[4]],  [pos[2],pos[3],pos[7],pos[6]], 
                [pos[1],pos[3],pos[7],pos[5]],  [pos[4],pos[6],pos[2],pos[0]]]
                
        verts = [[unit_cell[0],unit_cell[1],unit_cell[3],unit_cell[2]], [unit_cell[4],unit_cell[5],unit_cell[7],unit_cell[6]], 
                [unit_cell[0],unit_cell[1],unit_cell[5],unit_cell[4]],  [unit_cell[2],unit_cell[3],unit_cell[7],unit_cell[6]], 
                [unit_cell[1],unit_cell[3],unit_cell[7],unit_cell[5]],  [unit_cell[4],unit_cell[6],unit_cell[2],unit_cell[0]]]

        self.Crystalax.cla()
        if self.Edge_check.isChecked(): lw_ = 2
        else: lw_ = 0
        if self.Face_check.isChecked(): alpha_ = 0.25
        else: alpha_ = 0
        self.verts = self.Crystalax.add_collection3d(Poly3DCollection(verts, facecolors = self.crysface_color, linewidths=lw_, edgecolors=self.crysedge_color, alpha=alpha_))
        self.Crystalax.set_xlabel('a')
        self.Crystalax.set_ylabel('b')
        self.Crystalax.set_zlabel('c')
        for i in range(len(pos_add)):
            edge_atoms_x = False
            edge_atoms_y = False
            edge_atoms_z = False
            res = self.set_pos(pos_add[i][0], pos_add[i][1], pos_add[i][2])
            x.append(res[0])
            y.append(res[1])
            z.append(res[2])
            bk_color = self.AddAtoms_entry_pos_y[i+1].palette().color(self.AddAtoms_entry_pos_y[i+1].backgroundRole())
            colors.append(bk_color.name())
            sizes.append(self.sizes_init[i+1]*mult)
            if self.Extended_cells_check.isChecked():
                for _ii in range(2):
                    for _jj in range(2):
                        for _kk in range(2):
                            if _ii != 0 or _jj != 0 or _kk != 0:
                                res = self.set_pos(pos_add[i][0]+_ii, pos_add[i][1]+_jj, pos_add[i][2]+_kk)
                                x.append(res[0])
                                y.append(res[1])
                                z.append(res[2])
                                bk_color = self.AddAtoms_entry_pos_y[i+1].palette().color(self.AddAtoms_entry_pos_y[i+1].backgroundRole())
                                colors.append(bk_color.name())
                                sizes.append(self.sizes_init[i+1]*mult)
            if False:
                if self.Include_edge_atoms_check.isChecked():
                    if pos_add[i][0] == 0.0:
                        pos_x = 1
                        if self.Extended_cells_check.isChecked():
                            pos_x += 1
                        edge_atoms_x = True
                    elif pos_add[i][0] == 1.0:
                        pos_x = 0
                        edge_atoms_x = True
                    if pos_add[i][1] == 0.0:
                        pos_y = 1
                        if self.Extended_cells_check.isChecked():
                            pos_y += 1
                        edge_atoms_y = True
                    elif pos_add[i][1] == 1.0:
                        pos_y = 0
                        edge_atoms_y = True
                    if pos_add[i][2] == 0.0:
                        pos_z = 1
                        if self.Extended_cells_check.isChecked():
                            pos_z += 1
                        edge_atoms_z = True
                    elif pos_add[i][0] == 1.0:
                        pos_z = 0
                        edge_atoms_z = True
                    if edge_atoms_x:
                        res = self.set_pos(pos_x, pos_add[i][1], pos_add[i][2])
                        x.append(res[0])
                        y.append(res[1])
                        z.append(res[2])
                        bk_color = self.AddAtoms_entry_pos_y[i+1].palette().color(self.AddAtoms_entry_pos_y[i+1].backgroundRole())
                        colors.append(bk_color.name())
                        sizes.append(self.sizes_init[i+1]*mult)
                        if edge_atoms_y:
                            res = self.set_pos(pos_x, pos_y, pos_add[i][2])
                            x.append(res[0])
                            y.append(res[1])
                            z.append(res[2])
                            bk_color = self.AddAtoms_entry_pos_y[i+1].palette().color(self.AddAtoms_entry_pos_y[i+1].backgroundRole())
                            colors.append(bk_color.name())
                            sizes.append(self.sizes_init[i+1]*mult)
                        elif edge_atoms_z:
                            res = self.set_pos(pos_x, pos_add[i][1], pos_z)
                            x.append(res[0])
                            y.append(res[1])
                            z.append(res[2])
                            bk_color = self.AddAtoms_entry_pos_y[i+1].palette().color(self.AddAtoms_entry_pos_y[i+1].backgroundRole())
                            colors.append(bk_color.name())
                            sizes.append(self.sizes_init[i+1]*mult)
                    if edge_atoms_y:
                        res = self.set_pos(pos_add[i][0], pos_y, pos_add[i][2])
                        x.append(res[0])
                        y.append(res[1])
                        z.append(res[2])
                        bk_color = self.AddAtoms_entry_pos_y[i+1].palette().color(self.AddAtoms_entry_pos_y[i+1].backgroundRole())
                        colors.append(bk_color.name())
                        sizes.append(self.sizes_init[i+1]*mult)
                        if edge_atoms_z:
                            res = self.set_pos(pos_add[i][0], pos_y, pos_z)
                            x.append(res[0])
                            y.append(res[1])
                            z.append(res[2])
                            bk_color = self.AddAtoms_entry_pos_y[i+1].palette().color(self.AddAtoms_entry_pos_y[i+1].backgroundRole())
                            colors.append(bk_color.name())
                            sizes.append(self.sizes_init[i+1]*mult)
                    if edge_atoms_z:
                        res = self.set_pos(pos_add[i][0], pos_add[i][1], pos_z)
                        x.append(res[0])
                        y.append(res[1])
                        z.append(res[2])
                        bk_color = self.AddAtoms_entry_pos_y[i+1].palette().color(self.AddAtoms_entry_pos_y[i+1].backgroundRole())
                        colors.append(bk_color.name())
                        sizes.append(self.sizes_init[i+1]*mult)
                               
            
        
        if self.Atoms_check.isChecked(): self.scatter = self.Crystalax.scatter3D (x, y, z, color=colors, s = sizes)
        #self.Crystalax.pbaspect = [1.0, 1.0, 1.5]
        self.Crystalax.set_box_aspect((1.0, 1.0, 1.0))

        self.Crystalax.set_xlim([-1,plotlimits])
        self.Crystalax.set_ylim([-1,plotlimits])
        self.Crystalax.set_zlim([-1,plotlimits])
        self.Crystalcanvas.draw()
        
    def set_pos(self,x,y,z): 
        """ this function calculates the x,y and z positions of the atoms in an unit cell, based in their fractional positions """
        a_su = self.LatticeParams_slider['a'].value()
        b_su = self.LatticeParams_slider['b'].value()
        c_su = self.LatticeParams_slider['c'].value()
        alpha_su = self.LatticeParams_slider['alpha'].value()
        beta_su = self.LatticeParams_slider['beta'].value()
        gamma_su = self.LatticeParams_slider['gamma'].value()
        sa = np.sin(alpha_su*self.degree)
        ca = np.cos(alpha_su*self.degree)
        sb = np.sin(beta_su*self.degree)
        cb = np.cos(beta_su*self.degree)
        sg = np.sin(gamma_su*self.degree)
        cg = np.cos(gamma_su*self.degree)
        abg = (ca-cg*cb)/sg
        return [x*a_su+y*b_su*cg+z*c_su*cb,y*b_su*sg+z*c_su*abg,z*c_su*np.sqrt(sb*sb-abg*abg)]

    def calculate_xpd(self, en, from_opts):
        """ function that calculates the powder diffration intensity, based on the structure factor, energy dependent form factors and a normalization with Q^-2 """
        self.intensity = np.zeros (len(self.tth_range))
        wvl = self.Wvl_slider.value()
        CrysSize = self.CrystalSize_slider.value()
        for hkl in self.list_of_hkl_used:
            Q = self.Qhkl(*hkl)
            self.intensity += self.calculate_F2(hkl, en, Q)*self.calculate_intensity(hkl, wvl, Q, CrysSize)/(Q*Q)
        if from_opts:
            self.main_plot.set_data(self.tth_range, self.intensity)
        else:
            self.main_plot.set_ydata(self.intensity)
        self.XPDcanvas.draw_idle()

    def update_limits(self):
        """ function that updates the list of HKL used for calculating the scattering pattern based on the limits of the Q Qmin and Qmax """
        self.list_of_hkl_used = []
        for _ii in self.list_of_hkl:
            Q = self.Qhkl(*_ii)
            if (Q > self.Qi) and (Q < self.Qf):
                self.list_of_hkl_used.append(_ii)

    def Qhkl(self,h,k,l):
        """ returns the Q value of a HKL peak"""
        a = self.LatticeParams_slider['a'].value()
        b = self.LatticeParams_slider['b'].value()
        c = self.LatticeParams_slider['c'].value()
        alpha = self.LatticeParams_slider['alpha'].value()
        beta = self.LatticeParams_slider['beta'].value()
        gamma = self.LatticeParams_slider['gamma'].value()
        ha = h/a
        kb = k/b
        lc = l/c
        sa = np.sin(alpha*self.degree)
        ca = np.cos(alpha*self.degree)
        sb = np.sin(beta*self.degree)
        cb = np.cos(beta*self.degree)
        sg = np.sin(gamma*self.degree)
        cg = np.cos(gamma*self.degree)
        return 2*np.pi*np.sqrt((ha*sa*ha*sa+ kb*sb*kb*sb+ lc*sg*lc*sg + 2*ha*kb*(ca*cb-cg)+ 2*ha*lc*(ca*cg-cb) + 2*kb*lc*(cb*cg-ca))/(1.-ca*ca-cb*cb-cg*cg+2*ca*cb*cg))

    def calculate_F2(self,hkl,en_, Q):
        """ Function which calculates the structure factor. Many approaches were tried in order to descrease the time of the calculation. The best was this last one. """
        en = self.E*1000
        if False: # this one here was not the best
            atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[0])))
            Fhkl = atom.f0(Q)+ atom.f1(en)+1j*atom.f2(en)
            for i in range (self.additional_atoms):
                 atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[i+1])))
                 f = atom.f0(Q)+atom.f1(en)+1j*atom.f2(en)
                 pos = [self.AddAtoms_slider_x[i+1].value(),self.AddAtoms_slider_y[i+1].value(),self.AddAtoms_slider_z[i+1].value()]
                 Fhkl += f*np.exp(-2*np.pi*1j*np.dot(hkl,pos))
            return np.power(np.abs(Fhkl),2)
        else:
            s = 'h{}k{}l{}'.format(*hkl)
            if en_:
                atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[0])))
                Fhkl = atom.f0(Q)+ atom.f1(en)+1j*atom.f2(en)
                self.Fhkl.update({s:Fhkl})
                self.f = {}
            else:
                try:
                    Fhkl = self.Fhkl[s]
                except:
                    atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[0])))
                    Fhkl = atom.f0(Q)+ atom.f1(en)+1j*atom.f2(en)
                    self.Fhkl.update({s:Fhkl})
            for i in range (self.additional_atoms):
                 s2 = 'h{}k{}l{}i{}'.format(*hkl,i)
                 if en_:
                    atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[i+1])))
                    f = atom.f0(Q)+atom.f1(en)+1j*atom.f2(en)
                    self.f.update({s2:f})
                 else:
                    try:
                        f = self.f[s2]
                    except:
                        atom = eval('xu.materials.elements.{}'.format(str(self.Atom_types[i+1])))
                        f = atom.f0(Q)+atom.f1(en)+1j*atom.f2(en)
                        self.f.update({s2:f})
                 pos = [self.AddAtoms_slider_x[i+1].value(),self.AddAtoms_slider_y[i+1].value(),self.AddAtoms_slider_z[i+1].value()]
                 Fhkl += f*np.exp(-2*np.pi*1j*np.dot(hkl,pos))
            abs = np.abs(Fhkl)
        return abs*abs

    def Q2tth(self,Q, wvl): 
        """ function that returns the tth of a given Q. """
        return 360./np.pi*np.arcsin(Q*wvl/(4*np.pi))
        
    def size_width(self,tth, wvl, CrysSize): 
        """ sherer formula to set the peak width."""
        return 0.9*wvl/(2.355*CrysSize*np.cos(tth/2.*self.degree))

    def calculate_intensity(self,hkl, wvl, Q, CrysSize): 
        """ calculates the intensity of a peak"""
        tth = self.Q2tth(Q, wvl)
        w = self.size_width(tth, wvl, CrysSize)*360/np.pi
        return 1./(np.sqrt(2*np.pi)*w)*np.exp(-(self.tth_range-tth)*(self.tth_range-tth)/(2.*w*w))

if __name__ == '__main__':
    default = Defaults()
    if not os.path.isfile("pxrd.defaults"): default.createDefault()
    app = QApplication(sys.argv)
    a = Icons()
    pxrd = Window(default)
    pxrd.show()
    sys.exit(app.exec_())
