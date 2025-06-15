# California Price Estimation

Projeto com interface web dedicada a prever valores de casas na Califórnia

Baseado no projeto de regressão do módulo 14 do curso de ciência de dados da Hashtag Treinamentos

[Link do projeto na cloud do streamlit](https://www.reddit.com/r/linuxquestions/comments/dbiban/exclude_directory_from_tree_command/)

Árvore do projeto

```
├── data
│   ├── california-counties.geojson	<- Dados geográficos dos condados da califórnia e suas geometrias
│   ├── cleaned_housing.parquet		<- Dados originais limpos
│   ├── counties.parquet		<- Dados agregados de cada condado
│   └── housing.csv.zip			<- Dados originais
├── home.py				<- Interface de usuário web
├── logs
├── models
│   └── ridge_polyfeat_target_quantile.joblib	<- Modelo treinado com o algoritmo de regressão Ridge, features polinomiais e transformadas, e target transformado
├── notebooks
│   ├── basic_checks.ipynb			<- Checagens iniciais dos dados originais
│   ├── bivariate_analisys.ipynb		<- Análise bivariada dos dados originais
│   ├── data_mining_elasticnet.ipynb		<- Mineração de dados com algoritmo ElasticNet
│   ├── data_mining.ipynb			<- Mineração de dados inicial com comparação entre o algoritmo de Regressão Linear e DummyRegressor
│   ├── data_mining_ridge.ipynb			<- Mineração de dados com algoritmo Ridge
│   ├── geo_analisys.ipynb			<- Análise dos dados geográficos
│   ├── optimizing.ipynb			<- Otimização dos dados limpos
│   ├── src
│   │   ├── config.py				<- Constantes com os caminhos dos arquivos importantes
│   │   ├── graphics.py				<- Funções gráficas
│   │   ├── models.py				<- Funções que envolvem modelos de regressão
│   │   └── utils.py				<- Funções úteis
│   └── univariate_analisys.ipynb		<- Análise univariada dos dados originais
├── README.md					
├── references
│   └── 01_dicionario_de_dados.md		<- Dicionário de dados da base original
└── requirements.txt				<- Arquivo de dependências
```
