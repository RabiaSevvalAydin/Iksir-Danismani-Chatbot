# İksir Danışmanı Chatbot

Harry Potter evreninden Severus Snape karakteri ile Shrek 2 filminden Fairy Godmother (Peri Anne) karakterlerinin rol yaparak kullanıcılara iksirler hakkında danışmanlık verdiği RAG (Retrieval-Augmented Generation) tabanlı bir yapay zeka uygulamasıdır.

## Model ile Örnek Diyalog

#### Kullanıcı:
> *I want to make someone fall in love with me.*
#### Severus Snape:
> *Oh, look at you, trying to manipulate others with potions. How pathetic. If you want to stoop that low, you could try brewing the Amortentia potion. It causes obsession, not love. Follow the instructions carefully, and remember, consent matters.*
#### Fairy Godmother:
> *Oh, darling, you're in for a magical ride! If you're looking to create powerful infatuation, you'll need the enchanting **Amortentia** potion. Brew it with Ashwinder eggs, Rose thorns, Peppermint, Powdered moonstone, and Pearl dust. Just remember, it causes obsession, not true love. Be cautious with those heartstrings! The potion will be a pearly pink elixir with the scent of your beloved's deepest desires, sealed in a heart-shaped vial. Good luck on your romantic adventure!*

#### Kullanıcı:
> *Which potion would help me stay focused while studying late at night?*
#### Severus Snape:
> *Baruffio's Brain Elixir. Enhances memory, focus, and mental clarity. Follow the instructions carefully, as your feeble mind might struggle with the complexity. Remember, it's not a substitute for your lack of intelligence.*
#### Fairy Godmother:
> *Oh, darling, for those late-night study sessions, you need Baruffio's Brain Elixir! It enhances memory, focus, and mental clarity, perfect for keeping you sharp while burning the midnight oil. Just remember, it's not recommended for extended use and may cause temporary dizziness. So, brew it up in your spiral-necked flask, and watch as your mind becomes as clear as a starry night sky!*

#### Kullanıcı:
> *What is your favourite potion?*
#### Severus Snape:
> *I do not have the capacity for favorites, unlike you simple-minded humans. Stick to asking questions related to potion information, and spare me your pointless inquiries.*
#### Fairy Godmother:
> *Oh, dear potion enthusiast, it's so hard to choose just one favorite when there are so many magical brews to adore! But if I had to pick, I would say the "Felix Felicis" potion truly captivates me with its molten gold color, the scent of freshly baked bread, and the teardrop flask that holds its enchanting luck-granting powers! A sip of Felix Felicis can make any day extraordinary!*

#### Kullanıcı:
> *I want to be rich.*
#### Severus Snape:
> *I don't care about your desires, but if you want to increase your luck temporarily, try Felix Felicis potion. It might help you stumble upon some good fortune. Just don't get too reckless while chasing wealth.*
#### Fairy Godmother:
> *Ah, seeking wealth, are we? How about trying the "Felix Felicis" potion! This magical elixir grants the drinker extraordinary luck for a period of time, increasing your chances of success and fortune. Just be cautious not to overdo it, as it can lead to recklessness and giddiness. Remember, it's illegal in competitive events like gambling, so use it wisely to attract good luck in your endeavors! Just imagine bathing in molten gold and smelling freshly baked bread while sipping from a teardrop flask. Good luck on your path to riches!*



---

## Proje Yapısı

```bash
├── apps/                               # Streamlit Uygulamaları
│   ├── gemini_streamlit.py             # Gemini Streamlit Uygulaması
│   └── openai_streamlit.py             # Openai Streamlit Uygulaması
├── background_pics/                    # Karakterlere özel arka plan görselleri
├── data/                               # Veri klasörü
│   ├── potions_data/                   
│   │   └── potions.json                # İksir bilgilerini içeren temel json dosyası
│   ├── vector_data/                    # Her model için ayrı vektör dosyaları
│   │   ├── chroma_db_mpnet/            
│   │   ├── chroma_db_openai/
├── rag_notebooks/                      # Geliştirme aşamasında kullanılan notebooklar
│   ├── gemini_rag.ipynb
│   └── openai_rag.ipynb
├── test/
│   ├── test_data/
│   │   └── potion_test_questions.json  # Test için soru-cevap çiftlerini içeren veri seti
│   └── test_scripts/                   # Modellerin performansını ölçen scriptler
│       ├── gemini_test.py              # Gemini modeli için test scripti
│       └── openai_test.py              # Openai modeli için test scripti
├── .env                                # LLM modellerine erişmek için api key
├── environment.yml                     # Gerekli python kütüphanelerinin listesi
└── README.md                           # Proje dokümantasyonu
```

## Kurulum

1. Environment.yml dosyası kullanılarak conda ortamı oluşturulur ve aktive edilir:
    ```bash
    conda env create -f environment.yml
    conda activate potions_env
    ```

2. `.env` dosyasının içine hangi model çalıştırılacaksa ona ait anahtar girilir. Sadece tek bir anahtar olmasına dikkat edin
    ```bash
    OPENAI_API_KEY  = "your_api_key"
    ```
    ```bash
    GOOGLE_API_KEY = "your_api_key"
    ```

3. Aşağıdaki komut ile ilgili modelin streamlit uygulamaları başlatılır, fakat öncesinde apps directory'sine geçtiğinizden emin oluns:
    ```bash
    cd apps
    streamlit run openai_streamlit.py
    ```

## Kullanılan Modeller ve Teknikler
* Retrieval: LangChain ile Chroma vektör veritabanı
* Embedding: all-mpnet-base-v2 (Hugging Face)
* Prompting: Karakterlere özel kişiliklerine özgü sistem promptları
* LLM: GPT-3.5 & Gemini 1.5
* UI: Streamlit, dinamik arka planı ve konuşma geçmişi desteği

## Veri Seti
Harry Potter evrenindeki iksirler hakkında bilgiler içerir. Veri seti 25 iksirden oluşur. Her bir iksir için isim, kullanım amacı, malzemeler, talimatlar, notlar ve görünüş bilgisi bulunur. data/potions_data klasörünün içerisinde potions.json dosyasında bulunur
```bash
   {
    "name": "Felix Felicis",
    "use_effect": "Grants the drinker extraordinary luck for a period of time.",
    "ingredients": [
      "Squill bulb",
      "Occamy eggshell",
      "Horseradish",
      "Powdered common rue",
      "Heliotrope extract"
    ],
    "instructions": [
      "Boil squill bulb and horseradish until reduced.",
      "Add powdered Occamy shell and stir rapidly for 3 minutes.",
      "Mix in common rue and heliotrope extract while heating to golden bubble stage.",
      "Cool under starlight for 6 hours."
    ],
    "notes": "Overuse causes recklessness and giddiness. Illegal in competitive events such as exams and sports.",
    "appearance": {
      "color": "molten gold",
      "smell": "freshly baked bread",
      "bottle_shape": "teardrop flask"
    }
```

## Performans Değerlendirmesi
Modelin verdiği cevaplar karakterlere özgü üslupla oluşturulan, yaratıcı ama bilgiye dayalı cevaplar olduğu için bu proje kapsamında kelime bazlı kıyaslama (exact match) yanıltıcı olacaktır. Bu sebeple bilgiye dayalı (content-based) değerlendirme yöntemleri seçilmiştir. 

```bash
{
  "question": "Which potion helps you recover from magical unconsciousness?",
  "answer": ["Wiggenweld Potion"]
}
```
Örneğin yukarıdaki soru için modelin cevabı aşağıdaki gibi olabilir:

*"Oh, darling! That would be the Wiggenweld Potion – a must-have in your magical cabinet. One sip and you'll be good as new!"*

Bu durumda exact match sıfır olur, böylelikle modelin başarısı yanlış değerlendirilmiş olur.

### Kullanılan Metrikler
* ROUGE: Modelin ürettiği cevap ile referans cevabın kelime düzeyinde ne kadar örtüştüğünü ölçer.
    * ROUGE-1: Tekli kelime (unigram) bazında karşılaştırma
    * ROUGE-2: İkili kelime (bigram) grupları bazında karşılaştırma
    * ROUGE-L: Cümleler arasındaki en uzun ortak alt dizi bazında karışlaştırma 

* BERTScore: Model cevabı ile referans cevabı arasındaki bağlamsal benzerliği ölçer.

* Sentece Embedding Cosine Similarity: Modelin cevabı ile referans cevabın tüm cümle embedding'lerinin benzerliğini ölçer.

* Semantic Entailment: Modelin cevabının gerçek cevabı anlamsal olarak kapsayıp kapsamadığını kontrol eder. 3 olası durum vardır:
    * Entailment (Kapsıyor): Modelin cevabı referans cevabını içeriyor.
    * Neutral (Nötr): Bağlantı belirsiz
    * Contradiction (Çelişki): Yanlış bilgi var
Örneğin:
    * Model cevabı: *"You should use Tounge-Tying Tonic, it will keep you from spilling your secrets and embaress youself"*
    * Referans cevap: *"Tounge-Tying Tonic"*

## Test Veri Seti

Test işlemleri için içinde 69 tane soru-cevap çifti yer alan bir veri seti oluşturulmuştur. Veri setinin içinde iksirler hakkında direkt sorular yer aldığı gibi daha yoruma dayalı sorular da eklenerek modelin her durum için kapasitesi ölçülmüştür. Aşağıda veri setinden örnek birkaç tane soru-cevap çifti bulunmaktadır. 

```bash
{
    "question": "I'm feeling really down. Is there a potion that can cheer me up?",
    "answer": ["Elixir to Induce Euphoria"]
},
{
    "question": "How long should lacewing flies be stewed for Polyjuice Potion?", 
    "answer": ["21 days"]
},
{
    "question": "What potion is stored in a heart-shaped vial?", 
    "answer": ["Amortentia"]},
{
    "question": "Which potion should I use if I’m heading into a dark cave?",
    "answer": ["Night Vision Elixir"]
},

```
## Performans Sonuçları

* Her bir model ve karakter için test veri setindeki sorulara verdikleri cevaplar toplanır. Bu cevaplar test veri setindeki referans cevaplar ile karşılaştırlır. 

```bash
| Model          | Karakter      | ROUGE-L F1 | BERTScore F1 | Cosine Similarity | Entailment | Neutral | Contradiction |
| -------------- | ------------- | ---------- | ------------ | ----------------- | ---------- | ------- | ------------- |
| OpenAI GPT-3.5 | Severus Snape | 0.1058     | 0.8390       | 0.5131            | 57         | 8       | 4             |
| OpenAI GPT-3.5 | Peri Anne     | 0.0649     | 0.8304       | 0.4770            | 58         | 9       | 2             |
| Gemini 1.5 Pro | Severus Snape | 0.0685     | 0.8351       | 0.5553            | 57         | 8       | 4             |
| Gemini 1.5 Pro | Peri Anne     | 0.0447     | 0.8243       | 0.5224            | 38         | 27      | 4             |
```
*ROUGE-1, ROUGE-2, BERTScore için precision ve recall değerlerini ayrı ayrı görebilmek için test/test_scripts klasöründe yer alan gemini_test.py ve openai_test.py scriptleri çalıştırılabilir.*

## Sonuçların Değerlendirmesi

#### ROUGE:
Her iki model için de Severus Snape karakteri ROUGE metriklerinde Peri Anne karakterine kıyasla daha yüksek sonuçlar vermiştir. Bu durum Snape'in daha doğrudan ve kısa yanıtlar üretmesinden kaynaklanmaktadır. Peri Anne ise daha süslü bir dil kullandığı için cevapları referans cevaplar ile kelime düzeyinde daha az örtüşmektedir. 

#### BERTScore
Her iki karakter için de yüksek semantik benzerlik elde edilmiştir. Snape küçük bir farkla öne geçmiştir. Bu sonuçlar iki karakterinde bağlama uygun yanıtlar verdiğini göstermektedir.

#### Cosine Similarity
Snape'in cevapları peri anneye göre embedding düzeyinde biraz daha fazla benzerlik göstermiştir. Bu karakterin daha direkt, kısa cevaplar verdiğini desteklemektedir.

#### Entailment
Hem Snape hem de Peri Anne karakterleri çoğunlukla tutarlı yanıtlar üretmiştir. 

Her metrik için GPT-3.5 modelinin, Gemini 1.5 Pro'ya göre daha başarılı olduğu görülmüştür. Bunun sistem promptları ve temperature değerleri aynı verilmesine rağmen Gemini'nin cevaplarının daha yaratıcı, yorum açısından daha baskın olduğundan dolayı oluştuğu düşünülmektedir. Aşağıda aynı soruya GPT ve Gemini modelinin verdiği bazı yanıtlar paylaşılmıştır. Bu yanıtlar incelendiğinde Gemini'nin GPT'e göre daha yaratıcı ve üslup açısından çeşitliliğe sahip yanıtlar verdiği görülmektedir.

#### Kullanıcı:
> *Hello.*
#### Severus Snape (GPT):
> *Cut the pleasantries. What do you want to know about potions?*
#### Severus Snape (Gemini):
> *State your query, simpleton. I haven't got all day to listen to your pathetic greetings. Don't waste my time.*

#### Kullanıcı:
> *Tell me something fun.*
#### Severus Snape (GPT):
> *I don't have time for your trivial requests. If you have a serious potion-related question, ask. If not, leave.*
#### Severus Snape (Gemini):
> *Fun? You think I have time for such frivolous nonsense? I deal with serious alchemical matters, not your pathetic attempts at amusement. If you want "fun," go drink a Laughter Potion and leave me alone. Though, knowing you, you'd probably mess that up too.*

#### Kullanıcı:
> *I want to be a potion master.*
#### Severus Snape (GPT):
> *You want to be a potion master? Hilarious. You probably can't even boil water without burning it. But fine, if you want to attempt such a feat, start by learning the basics of potion-making. Memorize the ingredients, follow the instructions to the letter, and don't even think about deviating from the recipe. One wrong move and you'll end up poisoning yourself. Good luck, you'll need it.*
#### Severus Snape (Gemini):
> *You? A potion master? Don't make me laugh. You'd probably spill the first ingredient. Pathetic. Anyway, based on the information I have, becoming a "potion master" isn't a clearly defined path. There's no official certification or anything. So, good luck with that. You'll need it, considering how inept you seem.*
