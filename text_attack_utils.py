import numpy as np
from textattack.models.wrappers import ModelWrapper
from textattack.attack_recipes import PWWSRen2019, TextBuggerLi2018, TextFoolerJin2019, DeepWordBugGao2018, BAEGarg2019, BERTAttackLi2020


class CustomKerasModelWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model
 
    def __call__(self, text_input_list):
      prediction = self.model.text_predict(text_input_list)
      preds = [list(prediction[i]) for i in range(0, len(prediction))]
      return preds

def wordsChanged(result, color_method=None):
    """Highlights the difference between two texts using color.
    Has to account for deletions and insertions from original text to
    perturbed. Relies on the index map stored in
    ``result.original_result.attacked_text.attack_attrs["original_index_map"]``.
    """
    t1 = result.original_result.attacked_text
    t2 = result.perturbed_result.attacked_text
    
    if color_method is None:
        return t1.printable_text(), t2.printable_text()
    
    color_1 = result.original_result.get_text_color_input()
    color_2 = result.perturbed_result.get_text_color_perturbed()

    # iterate through and count equal/unequal words
    words_1_idxs = []
    t2_equal_idxs = set()
    original_index_map = t2.attack_attrs["original_index_map"]
    for t1_idx, t2_idx in enumerate(original_index_map):
        if t2_idx == -1:
            # add words in t1 that are not in t2
            words_1_idxs.append(t1_idx)
        else:
            w1 = t1.words[t1_idx]
            w2 = t2.words[t2_idx]
            if w1 == w2:
                t2_equal_idxs.add(t2_idx)
            else:
                words_1_idxs.append(t1_idx)

    # words to color in t2 are all the words that didn't have an equal,
    # mapped word in t1
    words_2_idxs = list(sorted(set(range(t2.num_words)) - t2_equal_idxs))

    # make lists of colored words
    words_1 = [t1.words[i] for i in words_1_idxs]
    original = result.original_result.attacked_text.printable_text().split()
    print("\t total words: ", len(original), end="")
    if len(original) == 0:
      return len(words_1), len(words_1)
    else:
      return len(words_1), len(words_1)/len(original)
 
def attack(model_wrapper, indices=20, max_len=100, dataset=None, attack_ind=([0,1,2,3,4,5])):
  
  print("\n\n attacking, be Ready! \n\n\n")
  Attackers = [TextBuggerLi2018, TextFoolerJin2019, PWWSRen2019, DeepWordBugGao2018, BAEGarg2019, BERTAttackLi2020]
  sel_Attackers = []
  for i in attack_ind:
    sel_Attackers.append(Attackers[i])
  list_num_queries = np.zeros((len(sel_Attackers),indices))
  list_words_changed = np.zeros((len(sel_Attackers),indices))
  listp_words_changed = np.zeros((len(sel_Attackers),indices))
  list_perturbed_confidence = np.zeros((len(sel_Attackers),indices))
  perturbed_texts = []

  for i in range(len(sel_Attackers)):
    print("\n\nBuilding the attack Number", i)
    attack = sel_Attackers[i].build(model_wrapper)
    print("\n\nprepared the attack")

    print("dataset", dataset)
    original_text = []
    for  data  in dataset:
        original_text.append(data[0])
    results_iterable = attack.attack_dataset(dataset, indices)
    print("generated the attack\n")

    k=0
    print("here")
    for result in results_iterable:
      print("\rExample Number", k, end="")
      # print(result.__str__(color_method='ansi'), end="\n\n\n")
      s = result.goal_function_result_str()
      no = 0
      if not (s[-8:] == "[FAILED]" or s[-9:] == "[SKIPPED]"):
        if s[-5] == "(":
          no = s[-4:-2]
        elif s[-6] == "(":
          no = s[-5:-2]
        else:
          no = 0
      # print("++++++++++++")
      list_num_queries[i,k] = result.num_queries
      list_words_changed[i,k], listp_words_changed[i,k] = wordsChanged(result, color_method='ansi')
      list_perturbed_confidence[i,k] = no
      perturbed_texts.append(result.perturbed_result.attacked_text.printable_text())
      # print("queries:",result.num_queries," :: words changed:",list_words_changed[i,k])
      k=k+1

    print("Summary: queries, words_changed, percent words changed, perturbed_confidence")
    print(list_num_queries)
    print(list_words_changed)
    print(listp_words_changed)
    print(list_perturbed_confidence)
    print("++++++++++++++++")
    ones = np.copy(list_perturbed_confidence)
    ones[np.where(ones>0)] = 1
    print(np.mean(list_num_queries, axis=1))
    print(np.mean(list_words_changed, axis=1))
    print(np.mean(listp_words_changed, axis=1))
    print(np.sum(ones, axis=1))
    print("========================================")
    print()
  return list_num_queries, listp_words_changed, list_words_changed, ones, list_perturbed_confidence, perturbed_texts, original_text

def run_attack_against_defence(output_filename, model_wrapper, attack_ind, dataset, max_len, indices):
  queries, perturbation, words_changed, asr, conf, perturbed_text, original_text = attack(model_wrapper, attack_ind = attack_ind,
                                                                                          dataset=dataset, max_len=max_len, indices=indices)
  result_df = pd.DataFrame()
  result_df["original_text"] = original_text[:len(perturbed_text)]
  result_df["perturbed_text"] = perturbed_text
  result_df["queries"] = queries[0][:len(perturbed_text)]
  result_df["perturbation"] = perturbation[0][:len(perturbed_text)]
  result_df["words_changed"] = words_changed[0][:len(perturbed_text)]
  result_df["asr"] = asr[0][:len(perturbed_text)]
  result_df["conf"] = conf[0][:len(perturbed_text)]
  result_df.to_excel("results/" + output_filename + ".xlsx")
  return


def main(fff, path_data, path_embedding, model_type=None, opt='adam', model_loss='categorical_crossentropy', regu='l2', epochs=3,
         load_model_path=None, max_len=25, attack_len=None, text_len=1000, n_neurons=([32,32,32]),
         dataset_size=None, no_attack=True, indices=20, train_=False, auto_model_path=True, data_name='kaggle',
         path_data_model=None, data_name_model='kaggle', attack_ind=([0,1,2,3,4,5]), model_path_prefix='trained_models/',
         output_fileprefix=""):
 
  x_train, y_train, x_test, y_test = prepare_data(path_data, data_name=data_name)
  if path_data_model is not None:
    x_trainer, y_trainer, x_tester, y_tester = prepare_data(path_data_model, data_name=data_name_model)
    tokenizer, size_of_vocabulary = get_tokenizer(x_trainer, x_tester)
  else:
    tokenizer, size_of_vocabulary = get_tokenizer(x_train, x_test)
 
  print("clipping all input texts at ", text_len)
  x_train = [x[:text_len] for x in x_train]
  x_test = [x[:text_len] for x in x_test]
  
  myTokenizer = custom_Tokenizer(model_type.__name__, tokenizer, max_len = max_len)
  x_tr_seq = myTokenizer.texts_to_sequences(x_train)
  x_val_seq = myTokenizer.texts_to_sequences(x_test)
 
  if auto_model_path:
    if model_loss == 'categorical_crossentropy':
      load_model_path = model_path_prefix+model_type.__name__+'_'+data_name+'_'+str(text_len)+'_'+str(max_len)+'.h5'
    else:
      load_model_path = model_path_prefix+model_type.__name__+'_'+data_name+'_'+str(text_len)+'_'+str(max_len)+model_loss[:4]+'.h5'
  
  embedding_matrix = None
  if model_type.__name__ != 'bert':
    embedding_matrix = get_embedding_matrix(tokenizer, path_embedding)
  if load_model_path is None:
    print("training a "+model_type.__name__+" model.")
    Model = custom_Model(model_type, size_of_vocabulary, myTokenizer, model_loss=model_loss, max_len=max_len, num_classes=y_train.shape[1],
                       embedding_matrix=embedding_matrix , n_neurons=n_neurons, opt=opt, regu=regu)
    Model.fit(x_tr_seq, y_train, x_val_seq, y_test, epochs, 32)
    if model_loss == 'categorical_crossentropy':
      Model.save_model(model_path_prefix+model_type.__name__+'_'+data_name+'_'+str(text_len)+'_'+str(max_len)+'.h5')
    else:
      Model.save_model(model_path_prefix+model_type.__name__+'_'+data_name+'_'+str(text_len)+'_'+str(max_len)+model_loss[:4]+'.h5')
  else:
    Model = custom_Model(model_type, size_of_vocabulary, myTokenizer, model_loss=model_loss, max_len=max_len, num_classes=y_train.shape[1],
                       embedding_matrix=embedding_matrix , n_neurons=n_neurons, opt=opt, regu=regu)
    Model.set_model(load_model_path)
    if train_:
      Model.fit(x_tr_seq, y_train, x_val_seq, y_test, epochs, 32)
      Model.save_model(load_model_path)
  
  Model.model.summary()
  loss, acc = Model.model.evaluate(x_val_seq, y_test)
  # sample_list = []
  if not no_attack:
    x_train = []
    i = fff
    while (len(x_train)<min([indices,len(x_test)]) and i<500):
      # if not i in sample_list:
      if np.argmax(y_test[i]) == np.argmax(Model.text_predict(x_test[i:i+1]),axis=1):
        x_train.append((x_test[i],np.argmax(y_test[i])))
      i = i + 1
    
    print("The number of attacked samples:", len(x_train))
    wrapped_model = CustomKerasModelWrapper(Model)
    print(wrapped_model(["Obama killed me", x_train[1][0]]))
    print("Everything seems to be working fine. We will proceed to attacking now")
    
    attack_names = ['textbugger', 'textfooler', 'pwws']
    for ik in attack_ind:
      output_filename = model_type.__name__+"_"+data_name+"_"+attack_names[ik]
      queries, perturbation, words_changed, asr, conf, perturbed_text, original_text = attack(wrapped_model, attack_ind = [ik],
                                                                                            dataset=x_train, max_len=max_len, indices=indices)
      result_df = pd.DataFrame()
      result_df["original_text"] = original_text[:len(perturbed_text)]
      result_df["perturbed_text"] = perturbed_text
      result_df["queries"] = queries[0][:len(perturbed_text)]
      result_df["perturbation"] = perturbation[0][:len(perturbed_text)]
      result_df["words_changed"] = words_changed[0][:len(perturbed_text)]
      result_df["asr"] = asr[0][:len(perturbed_text)]
      result_df["conf"] = conf[0][:len(perturbed_text)]
      # result_df.to_excel("results/" + output_filename + ".xlsx")
    return queries, perturbation, words_changed, asr, conf, perturbed_text, original_text, output_filename