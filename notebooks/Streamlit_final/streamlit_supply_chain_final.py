"""
Created on Tue Dec 26 17:20:11 2023
@author: Marine Merle
"""

import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from scipy.sparse import csr_matrix


# MISE EN FORME DE LA SIDEBAR 

## Logo
from PIL import Image
logo_lien = "Trustpilot.png"
logo_image = Image.open(logo_lien)
resized_logo = logo_image.resize((100,50))
st.sidebar.image(resized_logo, width=150)

## Annotations
st.sidebar.title("Application de notation")
st.sidebar.write("Bienvenue sur notre application Streamlit.")

## Sommaire
sommaire = st.sidebar.radio("Sommaire", ["Introduction", "Dataset et Preprocessing", "Analyse de données", "Modèles", "A vous de noter !"])

for _ in range(3):
    st.sidebar.markdown("&nbsp;", unsafe_allow_html=True)

st.sidebar.markdown("**Margaux ELISSALDE**")
st.sidebar.markdown("**Marine MERLE**")



# CONTENU DES DIFFERENTES PAGES DU STREAMLIT

## PARTIE 1 - INTRODUCTION

def afficher_partie1():
    st.title("Introduction")

    # Logo trustpilot
    image11_lien =  "Partie1_img1.jpg"
    image11 = Image.open(image11_lien)
    resized_image11 = image11.resize((450,300))
    col1, col2, col3 = st.columns([1, 2, 1])
    col2.image(resized_image11, use_column_width=True)

    # Texte d'introduction
    st.write("""
    Les avis clients dans le monde du e-commerce font aujourd’hui entièrement partie des habitudes d’achat des consommateurs.\n  Ces derniers sont en effet le troisième canal de recherche d’information sur un produit ou un service et 70% des consommateurs consultent des sites spécialisés d’avis en ligne avant de passer à l’acte d’achat.\n
    Ces systèmes de recueil de notes offrent aux consommateurs une prise supplémentaire pour appréhender les biens et orienter leur choix ; ils s’étendent désormais à une très grande diversité de biens et services : cosmétiques, hôtels et restaurants, électroménager, appareils photos, mais aussi services bancaires, etc.\n
    Trustpilot est une des plateformes proposant ce type de service.\n 
    Les notes étant si importantes …. 
    """)
    green_hex = "#27AE60"
    st.write(
        f"### <span style='color:{green_hex};font-weight:bold;font-size:1.2em;'>Est-il possible de les prédire ?</span>",
        unsafe_allow_html=True
    )

## PARTIE 2 - DATASET ET PREPROCESSING

def afficher_partie2():
    st.title("Dataset et Preprocessing")

    # Texte partie 2
    st.write("""
    Un des membres du projet travaillant dans le secteur de la cosmétique, nous avons ciblé notre projet sur ce domaine afin de pouvoir analyser plus en détail les attentes des consommateurs.\n
    Afin de récupérer des avis clients, nous avons réalisé du Webscrapping, via Beautiful soup, sur le site de Trustpilot pour quatre grandes enseignes : [Sephora](https://fr.trustpilot.com/review/www.sephora.com/), [Marionnaud](https://fr.trustpilot.com/review/www.marionnaud.com/), [My Origins](https://fr.trustpilot.com/review/www.my-origines.com/) et [Nocibé](https://fr.trustpilot.com/review/www.nocibe.fr/).\n 
    Notre dataframe initial se présentait de la façon suivante :\n 
    """)

    # Dataframe brut
    image21_lien =  "Partie2_img1.jpg"
    image21 = Image.open(image21_lien)
    resized_image21 = image21.resize((1000,400))
    st.image(resized_image21, caption="Dataframe initial 'brut'", use_column_width=True)

    st.write("""
    Nos données avaient le format initial suivant :\n
    """) 

    # Df.info() du dataframe brut
    image22_lien =  "Partie2_img2.jpg"
    image22 = Image.open(image22_lien)
    resized_image22 = image22.resize((450,300))
    col1, col2, col3 = st.columns([1, 2, 1])
    col2.image(resized_image22, caption="df.info() du dataframe initial", use_column_width=True)

    st.write("""
    Nous avons donc dû réaliser du Préprocessing de données afin de pouvoir les rendre exploitables. \n Nous avons pour cela :\n 
    -	Analysé et traité les valeurs manquantes,\n 
    -	Vérifié l’absence de doublons,\n 
    -	Changé le type des variables le nécessitant,\n
    -	Calculé d'autres variables nous intéressant dans notre analyse.\n
    A l’issue de ce travail, le dataframe se présentait de la façon suivante :\n 
    """)
    
    # Dataframe preprocessed
    image23_lien =  "Partie2_img3.jpg"
    image23 = Image.open(image23_lien)
    resized_image23 = image23.resize((1000,400))
    st.image(resized_image23, caption="Dataframe 'preprocessed'",use_column_width=True)

    # Df.info() du dataframe preprocessed
    image24_lien =  "Partie2_img4.jpg"
    image24 = Image.open(image24_lien)
    resized_image24 = image24.resize((550,350))
    col1, col2, col3 = st.columns([1, 2, 1])
    col2.image(resized_image24, caption="df.info() du dataframe preprocessed", use_column_width=True)

## PARTIE 3 - ANALYSE DE DONNEES

def afficher_partie3():
    st.title("Analyse de données")
    
    # Texte partie 3

    with st.expander("1-Distribution des notes",expanded=False):
        st.write("### Repartition des notes et analyse des données scrappées")
        st.write("""
        Une première visualisation pouvant intéresser notre analyse est la proportion de chacune des notes par entreprises. \n 
        Afin de pouvoir analyser une tendance et une évolution temporelle, nous avons dû scrapper des nombres de pages différents selon les entreprises. \n
        En effet, des entreprises comme Sephora ont beaucoup moins d'avis et retours clients qu'une entreprise telle que My Origins (qui ne vend qu'en ligne). \n
        La figure suivante représente ainsi le nombre d'avis client scrappés par entreprises.
        """)

        # Repartition du nombre d avis clients
        image31_lien =  "Partie3_img1.jpg"
        image31 = Image.open(image31_lien)
        resized_image31 = image31.resize((500,300))
        #col1, col2, col3 = st.columns([1, 2, 1])
        st.image(image31, caption="Répartition du nombre d'avis clients scrappés par entreprises", use_column_width=True)

        st.write("""
        Cette précision apportée, nous avons analysé la répartition des notes par entreprises.\n
        La figure suivante représente cette répartition.\n
        Bien que le volume de données scrappées ne soit pas comparable pour les quatre entreprises, nous notons que les entreprises Marionnaud ou
        My Origins ont de très bonnes notes (beaucoup de 5 et peu de 1). En revanche, les notes des entreprises Sephora ou Nocibé sont beaucoup plus mitigées voire mauvaises (beaucoup de 1).\n
        """)

        # Repartition des notes par entreprises
        image32_lien =  "Partie3_img2.jpg"
        image32 = Image.open(image32_lien)
        resized_image32 = image32.resize((500,300))
        #col1, col2, col3 = st.columns([1, 2, 1])
        st.image(image32, caption="Répartition des notes par entreprises",use_column_width=True)

        st.write("""
        Nous pouvons également noter que les clients sont extrêmes dans leurs notations : ces derniers sont soit très contents (note 5), ou très mécontents (note 1).\n
        Les notes intermédaires (2 à 4) sont beaucoup moins utilisées.\n
        """)

    with st.expander("2-Evolution des notes",expanded=False):
        st.write("### Evolution des notes moyennes par entreprises")
        st.write("""
        Nous avons souhaité poursuivre notre analyse de données avec l'évolution des notes moyennes par entreprises en fonction des années.\n
        La figure suivante représente cette évolution.\n
        Parmis les entreprises ayant de mauvaises notes :\n
        -	Sephora : a connu une chute de ses notes entre 2012 et 2019 mais semble, depuis, avoir de meilleurs notes
        -	Nocibé : a de mauvaises notes depuis 2016 sans connaitre d'évolution majeures
        Parmis les entreprises ayant de très bonnes notes :\n
         -	Marionnaud : connait une évolution croissente de ses notes depuis 2021
         -  My Origins : malgré le volume de données scrappées, nous constatons que les notations ne commencent qu'en 2022.\n
         La note moyenne de 2.5 en 2022 ne doit pas forcément être interprétée comme une mauvaise note. En effet, le scrapping ne couvre pas toute l'année 2022 et la note est ainsi biaisée.\n
         Nous constatons cependant que cette entreprise à de très bonnes notes en 2023. 
        """)

        # Evolution des notes par entreprises
        image34_lien =  "Partie3_img4.jpg"
        image34 = Image.open(image34_lien)
        resized_image34 = image34.resize((500,300))
        #col1, col2, col3 = st.columns([1, 2, 1])
        st.image(image34, caption="Evolution des notes moyennes par entreprises",use_column_width=True)

    with st.expander("3-Service après vente",expanded=False):
        st.write("### Evolution du délai moyen de réponse du SAV")
        st.write("""
        Nous avons souhaité analyser l'évolution du délai moyen de réponse du SAV.\n 
        En effet, répondre aux mauvais commentaires et comprendre l'origine de la déception d'un client est important pour une entreprise.\n
        Un retour rapide donnera une image d'une entreprise à l'écoute et soucieuse de la satisfaction de ses clients.\n
        La figure suivante représente l'évolution du délai moyen de réponses du SAV aux avis clients (bons comme mauvais).\n 
        Sans grande surprise, les marques ayant les meilleures notes (Marionnaud et My Origins) répondent à leurs clients très rapidement (~ 2.5 jours).\n
        Nocibé met beaucoup plus de temps à répondre (entre 15 et 25 jours) mais répond aux clients.\n
        Sephora ne semble en revanche pas avoir de service clients.\n
        En revanche, nous ne voyons pas de lien entre l’évolution de notes par années et les délais de réponse des SAV.
        """)

        # Evolution du délai moyen de réponse du SAV
        image35_lien =  "Partie3_img5.jpg"
        image35 = Image.open(image35_lien)
        resized_image35 = image35.resize((500,300))
        #col1, col2, col3 = st.columns([1, 2, 1])
        st.image(image35, caption="Evolution du délai moyen de réponse du SAV",use_column_width=True)

    with st.expander("4-Commentaires clients",expanded=False):
        st.write("### Analyse des sentiments")
        st.write("""
        Le constat des notes étant fait, il nous a semblé important de comprendre les commentaires et d'en faire leur analyse.\n
        L'analyse de sentiments des clients aidera en effet les entreprises à améliorer leurs produits et leurs services.\n
        Nous avons donc représentés deux wordcloud pour chacune des entreprises : 
        -   un pour les avis négatifs,
        -   un pour les avis positifs.
        """)

        # Wordcloud des avis negatifs par entreprise
        image36_lien =  "Partie3_img6.jpg"
        image36 = Image.open(image36_lien)
        resized_image36 = image36.resize((500,300))
        st.image(image36, caption="Wordcloud des avis negatifs par entreprise",use_column_width=True)

        st.write("""
        Concernant les avis négatifs, nous remarquons que les mots qui reviennent souvent sont « service client » et « colis ».\n
        Il semblerait donc qu’une insatisfaction est liée à l’envoi des colis et que les consommateurs font systématiquement appel au service client pour le problème en question.\n
        A préciser, que nous ne pouvons pas savoir si le service client dans ces cas est défaillant ou non.\n
        A noter que pour Nocibé, le mot « magasin » apparait également, il peut signifier un problème au niveau de l’expérience client en magasin.
        """)
        
        # Wordcloud des avis positifs par entreprise
        image37_lien =  "Partie3_img7.jpg"
        image37 = Image.open(image37_lien)
        resized_image37 = image37.resize((500,300))
        st.image(image37, caption="Wordcloud des avis positifs par entreprise",use_column_width=True)

        st.write("""
        Contrairement aux avis négatifs, aucun point commun ne semble se dégager pour les commentaires positifs : 
        -   Pour Séphora - les vendeur/euses, leurs conseils, leurs attitudes semblent être appréciés
        -   Pour Marionnaud – les produits et leurs efficacités : le packaging, les flacons, la qualité des produits sont mis en avant
        -   Pour Nocibé - aucun champ lexical ciblé ne ressort du wordcloud
        -   Pour My Origines (site de vente en ligne uniquement) - la rapidité, la conformité des produits, la livraison ainsi que le prix semblent satisfaire les clients
        A noter que My Origines vend des produits de luxe en ligne avec des réductions systématiques, il semblerait que ce point soit important pour le consommateur.
        """)

        st.write("""
        Dans la continuité de cette analyse, nous avons souhaité analyser les mots revenant le plus fréquemment dans les commentaires (positifs comme négatifs).\n
        """)

        # 15 mots les plus fréquents - avis négatifs et positifs
        image38_lien =  "Partie3_img8.jpg"
        image38 = Image.open(image38_lien)
        resized_image38 = image38.resize((500,300))
        image39_lien =  "Partie3_img9.jpg"
        image39 = Image.open(image39_lien)
        resized_image39 = image39.resize((500,300))
        st.image(image38, caption="15 mots les plus fréquents dans les avis clients négatifs",use_column_width=True)
        st.image(image39, caption="15 mots les plus fréquents dans les avis clients positifs",use_column_width=True)

        st.write("""
        Concernant les 15 mots qui reviennent les plus dans les commentaires, nous remarquons que :
        -	Pour les avis négatifs, ils sont liés à des problèmes de livraison de colis avec sollicitation du service client
        -	Pour les avis positifs, ils sont liés au prix des produits ainsi qu’à une livraison rapide, conforme et une expérience client réussie.\n
        """)


## PARTIE 4 - MODELES

def afficher_partie4():
    st.title("Modèles")
   
    st.write("""
    Comme nous venons de le voir, les notes attribuées à une entreprise dépend des commentaires laissés.\n
    En effet, les notes très mauvaises (1) intègrent dans leurs commentaires associés certains mots clefs (tels que 'colis jamais reçu', 'problème de livraison' etc. \n
    Afin de pouvoir prédire les notes mises par les clients en fonction de leurs commentaires, nous avons utlisé différents algorithmes de machine learning et un de deep learning.\n
    """)

    st.write("""
    Ces algorithmes étant relativement longs à entrainer, nous avons réduit notre jeu de données composé initialement de ~6000 données à ~1000 données.\n
    Nous avons veillé dans un premier temps à respecter, dans notre échantillon, la proportion dans la répartition de nos notes.\n
    Les notes 2, 3 et 4 étant cependant sous représentées, nos modèles ne les prédisaient pas bien. Nous avons donc réalisé un SMOTE afin d'améliorer les performances de nos algorithmes.\n
    Nous avons également entrainé des grid search afin de déterminer les meilleurs hyperparamètres de chacun des modèles.\n
    Seuls les résultats des trois meilleurs algorithmes seront présentés ici : \n
    -   (ML) Random Forest après Smote
    -   (ML) SVC après Smote
    -   (DL) Embedding
    Les résultats sont ici ceux obtenus sur le jeu de données complet.
    """)

    ## CHARGEMENT DES DONNEES

    df=pd.read_csv('Compilation webscrapping cosmetique.csv')

    df['notes']=df['notes'].astype('int')
    
    # On ne garde de notre dataframe que les commentaires (variable explicative) et les notes (variable à prédire)
    X, y = df.commentaire, df.notes

    # Séparation des données d'entrainement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Chargement du vectorizer enregistré
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)  

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

   
   ## RANDOM FOREST SUR JEU COMPLET APRES SMOTE

    st.subheader("Présentation du Random Forest sur jeu complet et après Smote")

    # Chargement du modèle Random Forest enregistré
    with open('random_forest_model.pkl', 'rb') as file:
        random_forest_model = pickle.load(file)

    # Afficher les meilleurs paramètres du Random Forest
    st.write("Les meilleurs paramètres du Random Forest sont :\n", random_forest_model.best_params_)

    y_pred_rfc_train = random_forest_model.predict(X_train)
    y_pred_rfc_test = random_forest_model.predict(X_test)

    # Afficher le classification report du modèle rfc sur les données d'entrainement
    st.write("Le classification report du modèle rfc sur les données d'entrainement est :\n", classification_report(y_train, y_pred_rfc_train))

    # Afficher le classification report du modèle rfc sur les données de test
    st.write("Le classification report du modèle rfc sur les données de test est :\n", classification_report(y_test, y_pred_rfc_test))

    # Afficher la matrice de confusion sur l'ensemble de test
    st.write("La matrice de confusion sur l'ensemble de test est :\n", pd.crosstab(y_test, y_pred_rfc_test)) 


    ## SVC SUR JEU COMPLET APRES SMOTE

    st.subheader("Présentation du SVC sur jeu complet et après Smote")

    # Chargement du modèle SVC enregistré
    with open('svc_model.pkl', 'rb') as file:
        svc_model = pickle.load(file)

    # Afficher les meilleurs paramètres du SVC
    st.write("Les meilleurs paramètres du SVC sont :\n", svc_model.best_params_)

    y_pred_svc_train = svc_model.predict(X_train)
    y_pred_svc_test = svc_model.predict(X_test)
    
    # Afficher le classification report du modèle SVC sur les données d'entrainement
    st.write("Le classification report du modèle SVC sur les données d'entrainement est :\n", classification_report(y_train, y_pred_svc_train))

    # Afficher le classification report du modèle SVC sur les données de test
    st.write("Le classification report du modèle SVC sur les données de test est :\n", classification_report(y_test, y_pred_svc_test))

    # Afficher la matrice de confusion sur l'ensemble de test
    st.write("La matrice de confusion sur l'ensemble de test est :\n", pd.crosstab(y_test, y_pred_svc_test)) 

    ## EMBEDDING SUR JEU COMPLET

    st.subheader("Présentation de l'Embedding sur jeu complet")

    max_len = 100 

    # Chargement du tokenizer enregistré
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

    X_train_text = tokenizer.texts_to_sequences(X_train)
    X_test_text = tokenizer.texts_to_sequences(X_test)

    train_text = pad_sequences(X_train_text, maxlen=max_len)
    test_text = pad_sequences(X_test_text, maxlen=max_len)


    # Chargement du modèle d'embedding enregistré
    embedding_model = load_model('C:/Users/Administrateur/Documents/Projet streamlit') 

    # Afficher le summary du modèle d'Embedding
    st.write("Le summary du modèle d'Embedding est :\n", embedding_model.summary()) 
    
    y_train_cat = to_categorical(y_train.cat.codes)
    y_test_cat = to_categorical(y_test.cat.codes)

    accuracy = embedding_model.evaluate(test_text, y_test_cat)[1]

    # Afficher l'accuracy 
    st.write("L accuracy sur notre jeu de test est : ", accuracy)



## PARTIE 5 - A VOUS DE NOTER

def afficher_partie5():
    st.title("A vous de noter !")
    st.write("Et vous ? Quelle note mettriez vous sur le produit récemment reçu ? ")
            
    # Chargement des modèles enregistrés
    with open('random_forest_model.pkl', 'rb') as file:
        random_forest_model = pickle.load(file)

    with open('svc_model.pkl', 'rb') as file:
        svc_model = pickle.load(file)

    with open('embedding_model.pkl', 'rb') as file:
        embedding_model = pickle.load(file)

    # Chargement du tokenizer
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

    max_words = 1000
    max_len = 100

    # Prédiction de la note à partir du commentaire et du modèle choisi
    def prediction_note(commentaire, model):
        if model == 'Random Forest':
            return random_forest_model.predict([commentaire])[0]
        elif model == 'SVC':
            return svc_model.predict([commentaire])[0]
        elif model == 'Embedding':
            return embedding_model.predict([commentaire])[0]

    # Champ de saisie du commentaire client
    commentaire_input = st.text_area("Saisissez votre avis ici:")

    # Sélection du modèle à utiliser
    model = st.selectbox("Choisissez le modèle à utiliser:", ['Random Forest', 'SVC', 'Embedding'])

    # Bouton pour déclencher la prédiction
    if st.button("Prédire"):
        if commentaire_input:
            # Prétraitement du commentaire
            commentaire_processed = tokenizer.texts_to_sequences([commentaire_input])
            commentaire_processed = pad_sequences(commentaire_processed, maxlen=max_len)
            predicted_note = prediction_note(commentaire_processed, model)
            st.success(f"La note prédite pour le commentaire est : {predicted_note}")
        else:
            st.warning("Veuillez saisir un commentaire avant de prédire.")


# Affichage de la page sélectionnée
if sommaire == "Introduction":
    afficher_partie1()
elif sommaire == "Dataset et Preprocessing":
    afficher_partie2()
elif sommaire == "Analyse de données":
    afficher_partie3()
elif sommaire == "Modèles":
    afficher_partie4()
elif sommaire == "A vous de noter !":
    afficher_partie5()