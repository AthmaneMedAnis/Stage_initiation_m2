import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import string

class PARHAFPreprocessor:
    def __init__(self):
        self.re_tabac = (
            r"\btabac\b|\btabagisme|\btabagique|intoxication tabagique|"
            r"(alcoolo|éthylo|ethylo)[- ]?tabagi|post[- ]?tabagi|"
            r"\bfumeu(r|se)s?\b|\bnon[- ]?fumeu(r|se)|\b(ancien(ne)?|ex)[- ]?fumeu(r|se)|"
            r"jamais\s+fum|\bfum(e|ait|aient|er|é)\b|\bcigarettes?\b|\bcig\b|\bcigares?\b|"
            r"cigarettes? électronique|\bvapot|e-cigarette|\bla pipe\b|nicotin|"
            r"\d\s*PA\b|\bpaquets?\b|paquets?[- ]?ann[ée]es?|\bsevr|sevrage tabagique|tabacolog"
        )

    def explode_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Mauvais type d'argument en entrée (DataFrame Pandas)")
        
        df_temp = df.copy()
        
        col = "documents"
        cols_to_explode = ["type", "header", "text", "word_count"]
        
        df_temp = pd.concat([
            df.drop(columns=col),
            pd.json_normalize(df[col])],
        axis = 1)
        
        mask = ((df_temp["type"].apply(len)) > 1)
        
        df_temp["_suffix"] = df_temp[mask]["type"].apply(
            lambda x: np.arange(1, len(x) + 1)
        )
        df_temp = df_temp.explode(cols_to_explode + ["_suffix"], ignore_index=True)
        
        mask = df_temp["_suffix"].notna()
        
        suffix = "_" + df_temp["_suffix"].astype(str)
        
        df_temp.loc[mask, "id"] = df_temp.loc[mask, "id"].astype(str) + suffix 
        
        df_temp = df_temp.drop(columns="_suffix")
        
        df_temp['word_count'] = df_temp['word_count'].astype(int)
        
        return df_temp
    
    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df_temp = df.copy()
        
        # suppression de la ponctuation puis des espaces
        df_temp['text'] = df_temp['text'].astype(str).str.replace(rf'[{string.punctuation}]', ' ', regex=True)
        df_temp['text'] = df_temp['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        # recalcul de la colonne word_count + securite au cas ou un texte est vide
        df_temp['word_count'] = df_temp['text'].str.split().str.len()
        df_temp['word_count'] = df_temp['word_count'].fillna(0).astype(int)
        
        return df_temp
    
    def stats(self, df:pd.DataFrame):
        
        print("Statistiques des word_count")
        print(df['word_count'].describe())
        
        print("\nRepartition par spécialité")
        print(df['specialty'].value_counts())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.histplot(df['word_count'], bins=50, kde=True, ax=axes[0], color='steelblue')
        axes[0].set_title('Distribution de la longueur des documents', fontsize=14)
        axes[0].set_xlabel('Nombre de mots', fontsize=12)
        axes[0].set_ylabel('Nombre de documents', fontsize=12)        
        axes[0].axvline(512, color='red', linestyle='--', label='Limite 512 mots')
        axes[0].legend()

        top_specialties = df['specialty'].value_counts().head(15)
        sns.barplot(x=top_specialties.values, y=top_specialties.index, hue=top_specialties.index, ax=axes[1], palette='viridis', legend=False)
        axes[1].set_title('Top 15 des spécialités médicales', fontsize=14)
        axes[1].set_xlabel('Nombre de documents', fontsize=12)
        axes[1].set_ylabel('')

        plt.tight_layout()
        plt.show()
        
    def detecter_mentions(self, df: pd.DataFrame) -> pd.DataFrame:
        df_temp = df.copy()
        
        df_temp['mention_tabac'] = df_temp['text'].str.contains(
            self.re_tabac, case=False, regex=True, na=False
        )
        
        nb_mentions = df_temp['mention_tabac'].sum()
        total = len(df_temp)
        print(f"Documents avec mention du tabac (selon les mots-clés de la doc des M1) : {nb_mentions} sur {total} ({(nb_mentions/total)*100:.1f}%)")
        
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(data=df_temp, x='mention_tabac', hue='mention_tabac', palette=['#ff9999', '#66b3ff'], legend=False)
        plt.title('Proportion des documents mentionnant le tabac', fontsize=12)
        plt.xlabel('Mention du tabac', fontsize=10)
        plt.ylabel('Nombre de documents', fontsize=10)
        plt.xticks(ticks=[0, 1], labels=['Non mentionné (False)', 'Mentionné (True)'])
        
        for container in ax.containers:
            ax.bar_label(container)
            
        ax.margins(y=0.15)
        plt.tight_layout()
        plt.show()
        
        return df_temp