import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

def get_entity_word(wiki2vec, words):
    entity_word={}
    #words = question.replace('?','').replace("'",' ').split()
    #print (words)
    for n in range(1,4):
        #print (i)
        entity_word[n]=[]
        for k in range(len(words)-n+1):
            #print ('k:',k,'n:',n)
            grams = words[k:k+n]
            if k == len(words)-n:
                next_token =''
            else:
                next_token = words[k+n]
            #print (grams)
            phrase = ' '.join(grams)
            if wiki2vec.get_word(phrase):
                #print('word', phrase)
                entity_word[n].append((phrase,next_token,'word'))
            elif wiki2vec.get_entity(phrase):
                #print('entity', phrase)
                entity_word[n].append((phrase,next_token,'entity'))
    #print (entity_word)
    for i in range(3,1,-1):
        #print (i)
        if len(entity_word[i])> 0:
            for entity in entity_word[i]:
                #print (entity)        
                for j in range(len(entity_word[i-1])):
                    phrase = entity_word[i-1][j]
                    nphr =phrase[0]+" "+phrase[1]
                    #print ("nphr: ",nphr)
                    if nphr in entity[0] or phrase[1] in entity[1]:
                        #print ("exchange ", entity_word[i-1][j], entity)
                        entity_word[i-1][j] = entity
    
    return entity_word[1]

def remove_stopwords(text):
    text= text.replace('?','')
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    #print(word_tokens) 
    return filtered_sentence 

def tokenized_sentence(wiki2vec, text):
    entity_word= get_entity_word(wiki2vec, remove_stopwords(text))
    return '|'.join(set([token[0] for token in entity_word]))

def tokenized_entities(wiki2vec, text):
    entity_word= get_entity_word(wiki2vec, remove_stopwords(text))
    return '|'.join([token[0] for token in entity_word if token[2]=='entity'])
