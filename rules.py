import matplotlib.pyplot as plt
import re
import seaborn as sns

from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report

def regle_de_base(texte):
    texte = str(texte).lower()
    # non-fumeur, jamais fumé
    if re.search(r'\b(non[- ]?fumeu(?:r|se)|jamais\s+fum)', texte):
        return "Non-fumeur"
    # ancien fumeur, ex-fumeur, sevré, sevrage tabagique, post-tabagique
    elif re.search(r'\b(?:ancien(?:ne)?|ex)[- ]?fumeu(?:r|se)|\bsevr|sevrage tabagique|post[- ]?tabagi', texte):
        return "Ancien fumeur"
    # intoxication tabagique, alcoolo-tabagique, PA (paquets-années), la pipe, vapot    
    # tabagisme, tabagique, fumeur, cigarette, cigare
    elif re.search(r'intoxication tabagique|(?:alcoolo|éthylo|ethylo)[- ]?tabagi|\d\s*PA\b|\bpaquets?\b|\bla pipe\b|\bvapot|e-cigarette|\btabagisme|\btabagique|\bfumeu(?:r|se)s?\b|\bcigarettes?\b|\bcigares?\b', texte):
        return "Fumeur actuel"
    else:
        return "Non mentionné"
    
    
def regle_avancee_contexte(texte):
    texte = str(texte).lower()
    
    regex_cible = r'\b(tabac|tabagisme|tabagique|fumeu(?:r|se)s?|cigarettes?|cigares?|cig|paquets?|vapot\w*|nicotin?|pa)\b'
    regex_negation = r'\b(pas|sans|non|aucun|aucune|jamais|nie|0)\b'
    regex_ancien = r'\b(ancien(?:ne)?|ex|arrêt(?:é)?|sevr\w*)\b'
    regex_famille = r'\b(père|mère|frère|soeur|parents|conjoint|mari|femme|famille)\b'
    
    mentions = list(re.finditer(regex_cible, texte))
    
    if not mentions:
        return "Non mentionné"
    
    # on analyse le contexte gauche de chaque mention 
    for match in mentions:
                
        # 30 caracteres juste avant et apres le mot-cle
        contexte_gauche = texte[max(0, match.start() - 30):match.start()].lower()
        contexte_droit = texte[match.end():min(len(texte), match.end() + 30)].lower()
        
        if re.search(regex_famille, contexte_gauche):
            continue
            
        if re.search(regex_ancien, contexte_gauche) or re.search(regex_ancien, contexte_droit):
            contexte_global = contexte_gauche + " " + contexte_droit
            
            if re.search(r'\b(tentative|échec)\b', contexte_global):
                pass
            elif "en cours" in contexte_droit[:15]: 
                return "Fumeur actuel"
            else:
                return "Ancien fumeur"
            
        if re.search(regex_negation, contexte_gauche) or re.search(regex_negation, contexte_droit[9]):
            return "Non-fumeur"
            
        return "Fumeur actuel"
        
    return "Non mentionné"

def evaluer_algorithme(y_true, y_pred, nom_algo, classes):
    print(f"{nom_algo} :")
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Coefficient de Kappa : {kappa:.4f}")
    
    if kappa < 0: accord = "Désaccord"
    elif kappa <= 0.2: accord = "Très faible"
    elif kappa <= 0.4: accord = "Faible"
    elif kappa <= 0.6: accord = "Modéré"
    elif kappa <= 0.8: accord = "Fort"
    else: accord = "Presque parfait"
    print(f"Niveau d'accord (Landis & Koch) : {accord}\n")
    
    print("Rapport de classification")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    
    matrice = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(matrice, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Matrice de Confusion : {nom_algo} vs Humain')
    plt.ylabel('Vérité Terrain (Humain)')
    plt.xlabel('Prédiction Algorithme')
    plt.show()
    print("\n\n")