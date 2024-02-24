import pandas as pd

def get_vocab(data: pd.DataFrame, person:str='Homer Simpson') -> list:
    person_phrases = data[(~data['spoken_words'].isna()) & (data['raw_character_text']==person)]
    person_phrases_list = list()
    for ph in person_phrases['spoken_words']:
        ph = ph.replace('"', '').replace('-', '').replace('/', '')
        if len(ph.split('[.?!]')) > 1:
            person_phrases_list.append(' '.join(ph.split()))
        ph = ph.split('[.?!]')
        ph = [' '.join(p.strip().split()) for p in ph] # Удалим лишние пробелы
        ph = [p  for p in ph  if len(p) > 1]
        person_phrases_list.extend(ph)
    return list(set(person_phrases_list))
