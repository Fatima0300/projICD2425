import streamlit as st
import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px
from wordcloud import WordCloud
import altair as alt
import geopy
from textblob import TextBlob
import seaborn as sns
from sklearn.decomposition import PCA
from collections import Counter
from streamlit import table
from sklearn.cluster import KMeans

# Ficheiros de dados
ficheiro = 'scopus.csv'
dados = pd.read_csv(ficheiro, encoding='utf-8')
scopus_normalizado = pd.read_csv('scopus_normalizado.csv', encoding='utf-8')

# Configurações da página
st.set_page_config(
    page_title="Dashboard ICD",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("Projeto Natural Language Processing")

# Menu lateral para navegação entre páginas
menu = st.sidebar.selectbox(
    "Menu",
    ["Introdução", "Análise Bibliométrica", "Modelação de Tópicos", "Clusterização", "Análises Adicionais"]
)

# Página de Introdução
if menu == "Introdução":
    st.markdown("""
    Neste dashboard são apresentados os resultados do projeto desenvolvido no âmbito da unidade curricular de Introdução à Ciência de Dados, do mestrado de Ciência de Dados para Ciências Sociais da Universidade de Aveiro.
    """)
    st.markdown(""" 
    O projeto teve como objetivo a análise de publicações científicas sobre a temática "Fatores Determinantes da Felicidade e Bem Estar Emocional", com recurso a técnicas de Processamento de Linguagem Natural (NLP). Numa primeira fase foi realizada uma análise bibliométrica das publicações obtidas e, posteriormente, procedeu-se a uma análise de conteúdo dos resumos dos artigos, tendo em vista a modelação de tópicos. 
    """)
    st.markdown("""
    Para visualizar os resultados, selecione a página desejada no menu lateral.
    """)
    st.markdown("""
    Para mais informações sobre o projeto, acesse o [repositório no GitHub](https://github.com/luizfrra/nlp_sentiment_analysis).
    """)

# Página de Análise Bibliométrica
elif menu == "Análise Bibliométrica":
    st.subheader("Análise Bibliométrica")
    st.markdown("""
    Nesta seção, são apresentados os resultados da análise bibliométrica das publicações obtidas.
    """)

    # Selectbox para selecionar a análise
    analise = st.selectbox(
        "Selecione a análise",
        ["Evolução Anual do Número de Publicações", "Mapa de Coocorrência de Termos", "Palavras-Chave mais Frequentes", "Top 10 Documentos mais Citados", "Top 10 Autores com mais Publicações", "Top 10 Instituições com mais Publicações"]
    )

    if analise == "Evolução Anual do Número de Publicações":
        st.markdown("### Evolução Anual do Número de Publicações")

        # Agrupar os dados por ano para calcular o número de publicações
        publicacoes_por_ano = dados.groupby('Year').size().reset_index(name='Número de Publicações')

        #Alterar o nome da coluna Year para Ano
        publicacoes_por_ano.rename(columns={'Year': 'Ano'}, inplace=True)

        # Exibir a tabela com os dados
        st.markdown("#### Tabela: Número de Publicações por Ano")
        st.dataframe(publicacoes_por_ano)  # Mostra a tabela no formato interativo  

        # Criar o gráfico de evolução anual do número de publicações com Plotly
        fig = px.line(
            publicacoes_por_ano, 
            x='Ano', 
            y='Número de Publicações', 
            labels={'Year': 'Ano de Publicação', 'Número de Publicações': 'Número de Publicações'},
            markers=True  # Adiciona marcadores nos pontos
        )

        # Melhorar a aparência do gráfico
        fig.update_traces(line=dict(color='green', width=2), marker=dict(color='purple', size=8, line=dict(width=2, color='purple')))
        fig.update_layout(
            template="plotly",  
            xaxis_title="Ano de Publicação",
            yaxis_title="Número de Publicações",
            showlegend=False
        )

        # Exibir o gráfico no Streamlit
        st.markdown("#### Gráfico: Evolução Anual do Número de Publicações")
        st.plotly_chart(fig)
    
    elif analise == "Mapa de Coocorrência de Termos":
        st.markdown("### Mapa de Coocorrência de Termos")
        st.markdown("""
        O mapa de coocorrência de termos é uma representação gráfica que permite visualizar a relação entre os termos mais frequentes nas publicações analisadas.
        """)

        # URL do VosViewer
        url = "https://tinyurl.com/22dwkz4j"
        
        # Exibir o mapa através de um IFrame"
        components.iframe(url, width = 1000, height = 600)
    
    elif analise == "Palavras-Chave mais Frequentes":
        st.markdown("### Palavras-Chave mais Frequentes")
        st.markdown("""
        Nesta seção, são apresentadas as palavras-chave mais frequentes nas publicações analisadas.
        """)

        # Fazer uma cópia da coluna 'Author Keywords' para preservar o DataFrame original
        author_keywords = dados['Author Keywords'].fillna('')  # Preencher valores ausentes na cópia

        # Excluir entradas vazias e espaços em branco
        author_keywords_cleaned = author_keywords[author_keywords.str.strip() != '']

        # Converter todas as palavras-chave para minúsculas e depois separar, contar e ordenar
        all_keywords = author_keywords_cleaned.str.lower().str.split('; ').explode().str.strip().value_counts().head(10)
    
        # Exibir a tabela com as palavras-chave mais frequentes
        st.markdown("#### Tabela: Palavras-Chave mais Frequentes")
        # Transformar 'all_keywords' (que é uma Series) em um DataFrame
        all_keywords_df = all_keywords.reset_index()

        # Renomear as colunas para 'Palavra-Chave' e 'Frequência'
        all_keywords_df.columns = ['Palavra-Chave', 'Frequência']

        # Exibir o DataFrame no Streamlit
        st.dataframe(all_keywords_df)

        # Criar um dataframe com as palavras-chave e as frequências
        palavras_frequentes = pd.DataFrame({'palavra': all_keywords.index, 'frequencia': all_keywords.values})

        # Criar o gráfico de barras horizontais com Plotly
        fig = px.bar(
            palavras_frequentes, 
            x='frequencia', 
            y='palavra', 
            orientation='h',  # barras horizontais
            title='Top 10 Palavras-Chave Mais Frequentes',
            labels={'frequencia': 'Frequência', 'palavra': 'Palavra-Chave'},
            color='frequencia',  # Cores para representar a frequência
            color_continuous_scale='Viridis'  # Escala de cor atraente
        )

        # Melhorar a aparência
        fig.update_layout(
            template="plotly",  # Visual agradável
            showlegend=False
        )

        # Exibir o gráfico no Streamlit
        st.plotly_chart(fig)

    elif analise == "Top 10 Documentos mais Citados":
        st.markdown("### Top 10 Documentos mais Citados")
        st.markdown("""
        Nesta seção, apresentamos os 10 documentos mais citados, com uma visualização em formato de linha do tempo. 
        Cada bolha representa um documento, onde o tamanho e a cor da bolha são proporcionais ao número de citações que o documento recebeu. 
        Ao passar o cursor sobre cada bolha pode ver mais detalhes, como o título, o autor e o número de citações.
        """)

        # Filtrar os 10 documentos mais citados
        top_cited_docs = dados.sort_values(by='Cited by', ascending=False).head(10)

        # Criar linha do tempo com bolhas proporcionais
        fig = px.scatter(
            top_cited_docs,
            x='Year',
            y='Title',
            size='Cited by',
            color='Cited by',
            hover_name='Title',
            hover_data={'Authors': True, 'Year': True, 'Cited by': True},  # Adicionar detalhes no hover
            title='Linha do Tempo: Documentos Mais Citados',
            labels={'Year': 'Ano de Publicação', 'Title': 'Documento', 'Cited by': 'Citações'},
            color_continuous_scale='Turbo'
        )

        # Melhorar a aparência
        fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(
            xaxis=dict(title='Ano de Publicação'),
            yaxis=dict(title='Título do Documento'),
            template="plotly_white"
        )

        # Exibir o gráfico no Streamlit
        st.plotly_chart(fig)

        # Iterar pelos 10 documentos mais citados e exibir as informações de forma estilizada
        for i, row in top_cited_docs.iterrows():
            st.markdown(f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #f9f9f9;">
                <h4 style="margin: 0; color: #333;">{row['Title']}</h4>
                <p style="margin: 5px 0;"><strong>Autores:</strong> {row['Authors']}</p>
                <p style="margin: 5px 0;"><strong>Ano de Publicação:</strong> {row['Year']}</p>
                <h3 style="margin: 5px 0; color: #1f77b4;">Citações: {row['Cited by']}</h3>
            </div>
            """, unsafe_allow_html=True)

    elif analise == "Top 10 Autores com mais Publicações":
        st.markdown("### Top 10 Autores com mais Publicações")
        st.markdown("""
        Nesta seção, apresentamos os 10 autores com mais publicações na base de dados.
        O gráfico de barras horizontais abaixo mostra a quantidade de publicações de cada autor.
        """)

        # Contar o número de publicações por autor
        top_authors = dados['Authors'].str.split('; ').explode().str.strip().value_counts().head(10)

        # Exibir a tabela com os autores com mais publicações
        st.markdown("#### Tabela: Autores com mais Publicações")
        # Transformar 'top_authors' (que é uma Series) em um DataFrame
        top_authors_df = top_authors.reset_index()

        # Renomear as colunas para 'Autor' e 'Publicações'
        top_authors_df.columns = ['Autor', 'Publicações']

        # Exibir o DataFrame no Streamlit
        st.dataframe(top_authors_df)

        # Criar um dataframe com os autores e o número de publicações
        autores_publicacoes = pd.DataFrame({'autor': top_authors.index, 'publicacoes': top_authors.values})

        # Criar o gráfico de barras horizontais com Plotly
        fig_bar = px.bar(
            autores_publicacoes, 
            x='publicacoes', 
            y='autor', 
            orientation='h',  # barras horizontais
            title='Top 10 Autores com mais Publicações',
            labels={'publicacoes': 'Publicações', 'autor': 'Autor'},
            color='publicacoes',  # Cores para representar o número de publicações
            color_continuous_scale='Viridis'  # Escala de cor atraente
        )

        # Melhorar a aparência
        fig_bar.update_layout(
            template="plotly",  # Visual agradável
            showlegend=False
        )

        # Exibir o gráfico no Streamlit
        st.plotly_chart(fig_bar)

        # Criar treemap após o gráfico de barras
        st.markdown("#### TreeMap dos Autores mais Produtivos")

        # Preparar dados para o treemap
        fig_tree = px.treemap(
            top_authors_df,
            path=['Autor'],
            values='Publicações',
            color='Publicações',
            color_continuous_scale='Viridis',
            title='Distribuição de Publicações por Autor'
        )

        # Melhorar layout
        fig_tree.update_layout(
            width=800,
            height=500,
            template="simple_white",
            paper_bgcolor='white',
            plot_bgcolor='white'

        )

        # Customizar texto
        fig_tree.update_traces(
            textinfo="label+value",
            hovertemplate='<b>%{label}</b><br>Publicações: %{value}<extra></extra>',
            marker=dict(        
                line=dict(width=2, color='white'),
                pattern=dict(shape="")
            ),
            root_color="white"
        )

        st.plotly_chart(fig_tree, use_container_width=True)

    elif analise == "Top 10 Instituições com mais Publicações":
        st.markdown("### Top 10 Instituições com mais Publicações")
        st.markdown("""
        Nesta seção, apresentamos as 10 instituições com maior número de publicações.
        O gráfico de barras horizontais abaixo mostra o número de publicações de cada instituição.
        """)

        # Contar o número de publicações por instituição
        top_instituicoes = dados['Affiliations'].str.split('; ').explode().str.strip().value_counts().head(10)

        # Exibir a tabela com as instituições com mais publicações
        st.markdown("#### Tabela: Instituições com mais Publicações")
        # Transformar 'top_instituicoes' (que é uma Series) em um DataFrame
        top_instituicoes_df = top_instituicoes.reset_index()

        # Renomear as colunas para 'Instituição' e 'Publicações'
        top_instituicoes_df.columns = ['Instituição', 'Publicações']

        # Exibir o DataFrame no Streamlit
        st.dataframe(top_instituicoes_df)

        # Criar um dataframe com as instituições e o número de publicações
        instituicoes_publicacoes = pd.DataFrame({'instituicao': top_instituicoes.index, 'publicacoes': top_instituicoes.values})

        # Criar o gráfico de barras horizontais com Plotly
        fig_bar = px.bar(
            instituicoes_publicacoes, 
            x='publicacoes', 
            y='instituicao', 
            orientation='h',  # barras horizontais
            title='Top 10 Instituições com mais Publicações',
            labels={'publicacoes': 'Publicações', 'instituicao': 'Instituição'},
            color='publicacoes',  # Cores para representar o número de publicações
            color_continuous_scale='Viridis'  # Escala de cor atraente
        )

        # Melhorar a aparência
        fig_bar.update_layout(
            template="plotly",  # Visual agradável
            showlegend=False
        )

        # Exibir o gráfico no Streamlit
        st.plotly_chart(fig_bar)

        # Supondo que o seu DataFrame se chame 'dados' e a coluna de afiliações seja 'Affiliations'
        # Vamos contar as 10 instituições com mais publicações
        instituicoes = dados['Affiliations'].str.split(';').explode().str.strip().value_counts().head(10)

        # Exemplo de mapeamento manual (você pode usar Geopy para automatizar isso)
        instituicoes_coords = {
            'Max Planck Institute for Human Development, Berlin, Germany': (52.46869, 13.30449),
            'Department of Psychology, Michigan State University, United States': (42.73104, -84.47378),
            'Escuela de Posgrado, Universidad Peruana Unión, Lima, Peru': (-11.98896, -76.83907),
            'Department of Human Development and Family Studies, Pennsylvania State University, University Park, United States': (40.79666, -77.85941),
            'Open University, Heerlen, Netherlands': (50.87845, 5.95902),
            'Department of Psychology, Faculty of Social Science and Liberal Arts, UCSI University, Kuala Lumpur, Malaysia': (3.12124, 101.65399),
            'Columbia University, New York, NY, United States': (40.80768, -73.96188),
            'Department of Psychological Science, University of California, Irvine, United States': (33.64761, -117.83872),
            'School of Psychology, University of Minho, Braga, Portugal': (41.56001, -8.39888),
            'Alexandru Ioan Cuza University, Iasi, Romania': (47.17454, 27.57342)
        }

        # Criar um dataframe com as instituições e suas coordenadas
        instituicoes_df = pd.DataFrame({
            'Instituição': instituicoes.index,
            'Publicações': instituicoes.values,
            'Latitude': [instituicoes_coords[i][0] for i in instituicoes.index],
            'Longitude': [instituicoes_coords[i][1] for i in instituicoes.index]
        })

        # Adicionando o título e a descrição no Streamlit
       
        st.markdown("""
        No seguinte mapa, mostramos as 10 instituições mais relevantes em termos de número de publicações científicas. 
        Cada ponto no mapa representa uma instituição, com o tamanho proporcional ao número de publicações, 
        facilitando a visualização das principais contribuições científicas por região geográfica.
        """)

        # Criar o mapa com Plotly
        fig = px.scatter_geo(
            instituicoes_df,
            lat='Latitude',
            lon='Longitude',
            size='Publicações',  # O tamanho dos pontos será proporcional ao número de publicações
            hover_name='Instituição',  # Ao passar o mouse, mostra o nome da instituição
            hover_data=['Publicações'],  # Mostrar também o número de publicações
            title='Top 10 Instituições com Mais Publicações',
            color='Publicações',  # Colorir os pontos de acordo com o número de publicações
            color_continuous_scale='Viridis',  # Escala de cor
            projection='natural earth',  # Projeção de mapa
            #template='plotly_dark'  # Tema para o gráfico
        )
        # Ajustar o layout do mapa
        fig.update_geos(
            showland=True,  # Mostrar a terra
            landcolor="white",  # Cor do fundo da terra
            showocean=True,  # Mostrar oceanos
            oceancolor="lightblue",  # Cor do oceano
            showlakes=True,  # Mostrar lagos
            lakecolor="lightblue",  # Cor dos lagos
            showcoastlines=True,  # Mostrar costas
            coastlinecolor="black",  # Cor das linhas costeiras
            showframe=True,  # Mostrar borda do mapa
            framecolor="black"  # Cor da borda do mapa
        )

        # Ajustar o tema para não ser escuro
        fig.update_layout(
            template=None,  # Remove o tema escuro
            paper_bgcolor="white",  # Cor do fundo do gráfico
            plot_bgcolor="white",  # Cor do fundo do mapa
            width=800,
            height=500)
        
        # Exibir o gráfico no Streamlit
        st.plotly_chart(fig)



# Página de Modelação de Tópicos
elif menu == "Modelação de Tópicos":
    st.subheader("Modelação de Tópicos")
    st.markdown("""
    Nesta seção, são apresentados os resultados da análise de conteúdo e da modelação de tópicos baseada em Processamento de Linguagem Natural (NLP).
    """)

    scopus_normalizado['Abstract_tokenized'] = scopus_normalizado['Abstract_Clean'].apply(lambda x: str(x).split() if isinstance(x, str) else [])

    # Criar Dicionário e Corpus
    dictionary = corpora.Dictionary(scopus_normalizado['Abstract_tokenized'])
    corpus = [dictionary.doc2bow(text) for text in scopus_normalizado['Abstract_tokenized']]
    
    # Definir o modelo LDA
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15, random_state=42, alpha='auto', per_word_topics=True)

    # Visualização do modelo LDA com PyLDAvis
    st.markdown("#### Visualização do Modelo LDA")

    # Preparar a visualização
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

    # Converter a visualização em HTML
    vis_html = pyLDAvis.prepared_data_to_html(vis)

    # Exibir a visualização no Streamlit
    components.html(vis_html, width=1300, height=800)

    # Escrever a descrição de cada tópico

    st.markdown("#### Descrição dos Tópicos")
    st.markdown("""
    A seguir, apresentamos uma descrição dos tópicos identificados pelo modelo LDA:
    """)
    # Obter as palavras-chave de cada tópico
    topics = lda_model.show_topics(formatted=False)

    # Dicionário com descrições dos tópicos
    descriptions = {
        0: "Este tópico está relacionado com questões psicológicas, sociais e de saúde que afetam o bem estar-emocional e a qualidade de vida dos indivíduos, abordando aspetos como a idade, estado civil, stress e relacionamentos, além de temas mais delicados como discriminação, abuso e questões no ambiente de trabalho.",
        1: "Este tópico está relacionado com questões de saúde mental e bem-estar emocional, especialmente no contexto de ansiedade, depressão e stress, destacando a experiência de jovens e o impacto destes fatores na sua vida. Além disso, aborda a importância da saúde mental e do apoio psicológico.",
        2: "Este tópico está relacionado com os desafios que surgem na interseção entre a vida profissional, familiar e pessoal, abordando questões como o impacto do trabalho no bem-estar e nas responsabilidades familiares. O tópico também explora o conflito entre o trabalho e a família, e como esse conflito pode afetar o equilíbrio entre vida profissional e pessoal.",
        3: "Este tópico está relacionado com o desenvolvimento pessoa e académico, explorando a interação entre o ambiente escolar e a personalidade dos indivíduos, bem como o impacto da educação e do ambiente escolar no bem-estar emocional e na qualidade de vida. Além disso, aborda a importância do apoio social e emocional no contexto educativo.",
        4: "Este tópico está centrado nas experiências de adolescentes no contexto educacional e social, abordando o impacto da escola no seu desenvolvimento e levando em conta aspectos como rendimento e religião, que podem influenciar as suas decisões e atitudes. O tópico também examina como as expectativas e resultados preditos sobre o futuro académico e profissional, como a universidade, se relacionam com a motivação dos adolescentes para alcançar os seus objetivos."
    }
    # Iterar pelos tópicos e exibir as palavras-chave
    for i, topic in topics:
        #st.markdown(f"**Tópico {i+1}:**")
        st.markdown(f"""
        <div style="background-color: #DEF2F1; padding: 10px; border-radius: 5px;">
            <h3 style="color: #478185;">Tópico {i+1}:</h3>
        </div>
        """, unsafe_allow_html=True)
        words = [word[0] for word in topic]
        st.markdown(f" - **Palavras-Chave:** {', '.join(words)}")
        st.markdown(f"**Descrição:** {descriptions[i]}")
        st.markdown("---")

# Página de Clusterização
elif menu == "Clusterização":
    st.subheader("Clusterização de Documentos com Base em Tópicos")
    
    st.markdown("""
    Nesta seção, são apresentados os resultados da clusterização de documentos baseada na distribuição de tópicos gerada pelo modelo LDA. Assim, a clusterização permite agrupar os documentos em clusters com base na similaridade das distribuições de tópicos.
    """)

    # Tokenização
    scopus_normalizado['Abstract_tokenized'] = scopus_normalizado['Abstract_Clean'].apply(lambda x: str(x).split() if isinstance(x, str) else [])

    # Criar Dicionário e Corpus
    dictionary = corpora.Dictionary(scopus_normalizado['Abstract_tokenized'])
    corpus = [dictionary.doc2bow(text) for text in scopus_normalizado['Abstract_tokenized']]

    # Definir o modelo LDA
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15, random_state=42, alpha='auto', per_word_topics=True)

    # Obter a distribuição de tópicos para cada documento
    topic_distributions = []
    for bow in corpus:
        topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0)
        topic_distributions.append([prob for _, prob in topic_distribution])

    # Converter para um array NumPy
    topic_matrix = np.array(topic_distributions)

    # Número ótimo de clusters
    k_optimal = 5
    st.write(f"Número de Clusters: {k_optimal}")

    # Clusterização com KMeans
    kmeans = KMeans(n_clusters=k_optimal, random_state=42)
    clusters = kmeans.fit_predict(topic_matrix)
    scopus_normalizado['Cluster'] = clusters

    # Cálculo da média das distribuições de tópicos por cluster
    df_reset = scopus_normalizado.reset_index(drop=True)
    topic_distributions_per_cluster = df_reset.groupby('Cluster').apply(
        lambda group: topic_matrix[group.index].mean(axis=0)
    )
    topic_distributions_per_cluster_df = pd.DataFrame(
        topic_distributions_per_cluster.tolist(),
        columns=[f'Topic {i+1}' for i in range(topic_matrix.shape[1])],
        index=[f'Cluster {i}' for i in range(k_optimal)]
    )

    # **Heatmap da Média das Distribuições de Tópicos por Cluster**
    st.subheader("Média das Distribuições de Tópicos por Cluster")

    # Criar o heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # A linha abaixo garante que a escala do heatmap seja proporcional aos valores das distribuições
    sns.heatmap(topic_distributions_per_cluster_df, annot=True, cmap='coolwarm', ax=ax, cbar=True)

    ax.set_title('Heatmap de Distribuições Médias de Tópicos por Cluster')

    st.pyplot(fig)

    # **Distribuição de Documentos por Cluster**
    st.subheader("Distribuição de Documentos por Cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    scopus_normalizado['Cluster'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title('Distribuição de Documentos por Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Número de Documentos')

    # Adicionar a frequência de palavras em cima de cada barra
    for i, count in enumerate(scopus_normalizado['Cluster'].value_counts().sort_index()):
        ax.text(i, count, str(count), ha='center', va='bottom')

    st.pyplot(fig)

    # **Palavras Mais Frequentes por Cluster**
    st.subheader("Palavras Mais Frequentes por Cluster")
    cluster_words = {i: [] for i in range(k_optimal)}

    # Analisar as palavras mais frequentes em cada cluster
    for i in range(k_optimal):
        cluster_df = scopus_normalizado[scopus_normalizado['Cluster'] == i]
        all_text = " ".join([" ".join(tokens) for tokens in cluster_df['Abstract_tokenized']])
        words = all_text.split()
        word_counts = Counter(words)
        most_common_words = word_counts.most_common(10)
        cluster_words[i] = [f"{word} ({freq})" for word, freq in most_common_words]

    # Criar um DataFrame para exibir
    max_len = max(len(words) for words in cluster_words.values())
    cluster_table = {f"Cluster {i}": cluster_words[i] + [""] * (max_len - len(cluster_words[i])) for i in cluster_words}

    results_df = pd.DataFrame(cluster_table)
    st.table(results_df)

    # **Visualização dos Clusters em 2D com PCA**
    st.subheader("Visualização dos Clusters em 2D com PCA")

    # Realizar PCA para reduzir a dimensionalidade
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(topic_matrix)

    # Plotar os clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=scopus_normalizado['Cluster'], cmap='viridis')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Visualização dos Clusters com PCA')

    # Adicionar uma legenda para os clusters
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    st.pyplot(fig)
    

# Página de Análises Adicionais
elif menu == "Análises Adicionais":
    st.subheader("Análises Adicionais")
    st.markdown("""Nesta seção, são apresentadas análises adicionais realizadas sobre os dados das publicações científicas.""")

    # Tokenização
    scopus_normalizado['Abstract_tokenized'] = scopus_normalizado['Abstract_Clean'].apply(lambda x: str(x).split() if isinstance(x, str) else [])

    # Criar Dicionário e Corpus
    dictionary = corpora.Dictionary(scopus_normalizado['Abstract_tokenized'])
    corpus = [dictionary.doc2bow(text) for text in scopus_normalizado['Abstract_tokenized']]

    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15, random_state=42, alpha='auto', per_word_topics=True)  
   
    # Obter as distribuições de tópicos para cada documento
    document_topic_distributions = [lda_model.get_document_topics(doc) for doc in corpus]

    # Determinar o tópico dominante para cada documento
    scopus_normalizado['Dominant_Topic'] = [max(topics, key=lambda x: x[1])[0] for topics in document_topic_distributions]

    # Exibir os primeiros resultados
    print(scopus_normalizado[['Abstract_Expanded', 'Dominant_Topic']].head())

    # Buscar documentos que têm como tópico dominante o tópico X
    def get_documents_by_topic(topic_id):
        return scopus_normalizado[scopus_normalizado['Dominant_Topic'] ==  topic_id][['Title', 'Dominant_Topic']]
    
    # Contar tópicos ao longo do tempo
    topic_trends = scopus_normalizado.groupby(['Year', 'Dominant_Topic']).size().unstack(fill_value=0)
    topic_trends.plot(kind='line', figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    topic_trends.plot(kind='line', ax=ax)
    ax.set_title('Tendência dos Tópicos ao Longo do Tempo')
    ax.set_xlabel('Ano de Publicação')
    ax.set_ylabel('Número de Documentos')
    ax.legend(title='Tópicos')
    st.pyplot(fig)

    # Análise de Sentimento
    st.markdown("### Análise de Sentimento")
    st.markdown("""A análise de sentimento é uma técnica de Processamento de Linguagem Natural (PLN) que permite identificar e extrair informações subjetivas de textos, como opiniões, sentimentos e emoções. Neste contexto, aplicamos a análise de sentimento aos resumos dos artigos científicos para identificar a polaridade dos sentimentos expressos.""")

    # Função para calcular o sentimento usando TextBlob
    def calcular_sentimento(text):
        if isinstance(text, str):
            return TextBlob(text).sentiment
        else:
            return None
    
    if 'Abstract_Expanded' not in scopus_normalizado.columns:
        st.error("A coluna 'Abstract_Expanded' é necessária para análise de sentimento.")
    else:
        scopus_normalizado['Sentiment'] = scopus_normalizado['Abstract_Expanded'].apply(calcular_sentimento)

        # Exibir os resultados
        if scopus_normalizado['Sentiment'].isnull().all():
            st.error("Nenhum texto válido foi encontrado para análise de sentimento.")
        else:
            # Etiquetar sentimentos com base na polaridade
            def label_sentiment(polarity):
                if polarity > 0:
                    return 'Positive'
                elif polarity < 0:
                    return 'Negative'
                else:
                    return 'Neutral'

            scopus_normalizado['Sentiment_Label'] = scopus_normalizado['Sentiment'].apply(lambda x: label_sentiment(x.polarity) if x else 'Neutral')

    # Contar o número de documentos por sentimento
    sentiment_counts = scopus_normalizado['Sentiment_Label'].value_counts()

    # Gráfico de barras dos sentimentos
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['lightblue', 'lightpink', 'lightgreen'])
    plt.title('Número de Documentos por Sentimento')
    plt.xlabel('Sentimento')
    plt.ylabel('Número de Documentos')
    for i, count in enumerate(sentiment_counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    st.pyplot(plt)

