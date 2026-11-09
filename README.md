# Relatório Final: Reconhecimento de Pokémon em Vídeo

**Disciplina:** Machine Learning e Deep Learning – Engenharia de Software

## Parte 5: Respostas e Análises dos Experimentos

### 1. Por que separar treino e validação por vídeo?
Eu testei treinar o modelo em dois cenários distintos. No primeiro, a separação foi feita de forma aleatória por fotos (frames misturados). No segundo, a separação ocorreu isolando vídeos inteiros.

Quando a divisão foi feita misturando os frames, a acurácia de validação atingiu rapidamente valores próximos a 99%. No entanto, ao testar no vídeo final (`teste_final.mp4`), o modelo apresentou muitas falhas. Quando separei por vídeo inteiro, a acurácia de validação foi menor (em torno de 93%), mas o modelo acertou muito mais no vídeo de teste real.

Isso ocorre devido ao "vazamento de dados" (*data leakage*). Um vídeo gravado a 30fps possui frames quase idênticos (o frame 100 e o 101 são visualmente a mesma coisa). Se o frame 100 cai no treino e o 101 na validação, o modelo está essencialmente "trapaceando", pois está sendo validado em uma imagem que ele já decorou no treino. A divisão por vídeo previne isso e obriga o modelo a generalizar. O número da acurácia gerado pelo split por vídeo representa o desempenho real de forma muito mais correta.

### 2. O fine-tuning ajudou?
Avaliando o modelo apenas treinando a camada Dense final e, em seguida, aplicando o Fine-tuning (descongelando as últimas 30 camadas da MobileNetV2), percebi uma diferença notável no gráfico de validação (`curvas.png`). Ao aplicar o Fine-tuning com a taxa baixa (`1e-5`), a Loss de validação continuou caindo suavemente e a precisão geral melhorou, pois o modelo adaptou os filtros de bordas e texturas genéricas da ImageNet para os formatos específicos dos Pokémons.

Porém, ao tentar o Fine-tuning com uma taxa alta (`1e-3`), o gráfico mostrou um pico na Loss de validação e a acurácia despencou. A taxa precisa ser extremamente baixa no Fine-tuning para evitar o "esquecimento catastrófico". Como os pesos da MobileNet já vieram bem treinados, passos grandes na descida do gradiente (`1e-3`) destroem esses pesos pré-existentes, destruindo o aprendizado básico.

### 3. O modelo reconhece o Pokémon de qualquer ângulo?
Retreinei o modelo removendo propositalmente todos os frames em que o Charmander aparece de costas. No teste do `detectar.py` com o vídeo do Charmander de costas, o modelo falhou. O modelo previu `?` (devido ao meu limiar de confiança) ou confundiu o Charmander com a classe fundo.

Isso prova que a IA baseia sua predição em características espaciais visíveis (como os olhos, barriga amarela e boca). Sem ver a silhueta de costas durante o treino, ela não tem referência matemática para aquela pose.

Duas estratégias que proponho para tornar o modelo mais robusto a ângulos novos são:
- **Data Augmentation agressivo:** Aplicar rotações e espelhamentos horizontais intensos durante o treino para simular posições inexistentes.
- **Enriquecer o dataset:** Gravar os vídeos garantindo a filmagem em 360 graus sob diferentes condições de iluminação (o que apliquei no meu script final).

### 4. O que acontece com um Pokémon que o modelo não conhece?
Apresentei ao vídeo de teste um Pokémon novo (ex: Squirtle). 

Sem minhas melhorias, a função *Softmax* forçaria a predição para um dos 3 Pokémons conhecidos (pois a soma das saídas deve ser sempre 100%). A rede diria, com 50% de certeza, que é um Bulbasaur.

Mas com a minha melhoria no `detectar.py` (Limiar de Confiança em 0.60 e Suavização de Janela), as probabilidades flutuaram (ficaram fracas e divididas) e a média caiu para menos de 60%. O código então exibe `?`. A classe "fundo" ajuda a capturar pixels genéricos e impede chutes errôneos caso não haja um alvo definido.

### 5. Classificação ou detecção?
Quando coloquei dois Pokémons lado a lado no mesmo quadro do vídeo, o script `detectar.py` começou a "piscar" rapidamente entre os nomes dos dois ou focou naquele que ocupava a maior área da imagem.

Isso acontece porque eu apliquei uma **Classificação de Imagem**, que analisa todos os pixels e fornece apenas um rótulo geral para a cena. 

Para resolver isso, eu precisaria usar **Detecção de Objetos** (como o modelo YOLOv8). 
- **Nos dados (anotação):** Eu precisaria desenhar *bounding boxes* ao redor de cada Pokémon e exportar as coordenadas (X e Y) correspondentes.
- **No modelo:** Teria que alterar a rede neural para prever múltiplas regiões de interesse simultaneamente em vez de usar uma camada global densa, o que permitiria encontrar e enquadrar os dois Pokémons independentemente num mesmo quadro.

---

### Linha do Tempo (Gerada pelo detectar.py)

```text
Linha do tempo (Pokemon identificados):
     0.0s a   19.0s  ->  bulbasaur
    19.0s a   36.9s  ->  pikachu
    37.2s a   38.0s  ->  charmander
    38.3s a   39.2s  ->  bulbasaur
    40.6s a   44.9s  ->  charmander
    44.9s a   54.4s  ->  charmander
Pokemon encontrados no video: ['bulbasaur', 'charmander', 'pikachu']
```