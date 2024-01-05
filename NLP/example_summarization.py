from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import torch

text = """summarize: Every day I get up at seven o'clock. I go to the bathroom and I have a shower. Then I put on my clothes. I usually eat cereal with milk for breakfast.
    I go to school by bus and arrive at a quarter past eight. School starts at half past eight and our lessons finish at half past one.
    I arrive home at two o'clock. I have lunch with my family.
    After lunch, I go to my room and do my homework. The time that I finish homework varies from day to day.
    Gone are the homework I play with my friends at the park in front my home and then I watch TV or play with my computer.
    I have dinner at half past eight. I usually help my mother in the kitchen. After dinner I usually watch TV and I relax,
    or read a magazine or something until 11pm. That's the time when I normally go to bed."""

text_ita = """L’azienda statunitense Broadcom, uno dei più grandi produttori di semiconduttori al mondo, ha presentato un’offerta per acquisire Qualcomm, altra grande società degli Stati Uniti conosciuta soprattutto per la sua produzione di microprocessori Snapdragon (ARM), utilizzati in centinaia di milioni di smartphone in giro per il mondo. Broadcom ha proposto di acquistare ogni azione di Qualcomm al prezzo di 70 dollari, per un valore complessivo di circa 105 miliardi di dollari (130 miliardi se si comprendono 25 miliardi di debiti netti) . Se l’operazione dovesse essere approvata, sarebbe una delle più grandi acquisizioni di sempre nella storia del settore tecnologico degli Stati Uniti. Broadcom ha perfezionato per mesi la sua proposta di acquisto e, secondo i media statunitensi, avrebbe già preso contatti con Qualcomm per trovare un accordo. Secondo gli analisti, Qualcomm potrebbe comunque opporsi all’acquisizione perché il prezzo offerto è di poco superiore a quello dell’attuale valore delle azioni dell’azienda. Ci potrebbero essere inoltre complicazioni sul piano dell’antitrust da valutare, prima di un’eventuale acquisizione."""
text_ita_2 = """Ogni giorno mi alzo alle sette. Vado in bagno e mi faccio la doccia. Poi mi vesto. Di solito mangio cereali con latte a colazione.
      Vado a scuola in autobus e arrivo alle otto e un quarto. La scuola inizia alle otto e mezza e le nostre lezioni finiscono all'una e mezza.
      Arrivo a casa alle due. Pranzo con la mia famiglia.
      Dopo pranzo vado in camera mia e faccio i compiti. Il tempo in cui finisco i compiti varia di giorno in giorno.
      Finiti i compiti, gioco con i miei amici nel parco davanti a casa e poi guardo la TV o gioco con il computer.
      Ho cenato alle otto e mezza. Di solito aiuto mia madre in cucina. Dopo cena di solito guardo la TV e mi rilasso,
      o leggere una rivista o qualcosa del genere fino alle 23:00. E' l'ora in cui normalmente vado a letto."""
T5_model_ita_news = (
    "it5/it5-base-news-summarization"  # modello specializzato per le news
)
T5_model_ita_base = "efederici/it5-base-summarization"
T5_model_eng = "T5-base"

tokenizer = AutoTokenizer.from_pretrained(T5_model_ita_base)

inputs = tokenizer(text_ita_2, return_tensors="pt")

model = AutoModelForSeq2SeqLM.from_pretrained(T5_model_ita_base)

outputs = model.generate(
    inputs["input_ids"],
    max_length=150,
    min_length=50,
    length_penalty=3.80,
    num_beams=4,
    # early_stopping=True,
)

print(
    tokenizer.decode(
        outputs[0], clean_up_tokenization_spaces=True, skip_special_tokens=True
    )
)
