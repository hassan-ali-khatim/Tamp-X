import json
from lime.explanation import DomainMapper
from IPython.display import display, HTML
import os
import string
from sklearn.utils import check_random_state
import re 
import numpy as np
import itertools
from tampering_utils import inverse_sigmoid, hard_sigmoid, sinusoidal_sigmoid, inference_model
from general_utils import load_trained_model, visualize_activations
from xai_utils import decode_sentence, Vectorizer
from sklearn.pipeline import make_pipeline

class TextDomainMapper(DomainMapper):
    """Maps feature ids to words or word-positions"""

    def __init__(self, indexed_string):
        """Initializer.
        Args:
            indexed_string: lime_text.IndexedString, original string
        """
        self.indexed_string = indexed_string

    def map_exp_ids(self, exp, positions=False):
        """Maps ids to words or word-position strings.
        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            positions: if True, also return word positions
        Returns:
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """
        if positions:
            exp = [('%s_%s' % (
                self.indexed_string.word(x[0]),
                '-'.join(
                    map(str,
                        self.indexed_string.string_position(x[0])))), x[1])
                   for x in exp]
        else:
            exp = [(self.indexed_string.word(x[0]), x[1]) for x in exp]
        return exp

    def visualize_instance_html(self, exp, label, div_name, exp_object_name,
                                text=True, opacity=True):
        """Adds text with highlighted words to visualization.
        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             text: if False, return empty
             opacity: if True, fade colors according to weight
        """
        if not text:
            return u''
        text = (self.indexed_string.raw_string()
                .encode('utf-8', 'xmlcharrefreplace').decode('utf-8'))
        text = re.sub(r'[<>&]', '|', text)

        # print("visualize_instance_html", exp)
        # print("======")
        exp_lst  = []
        for x in exp:
        #   print(x)
        #   print(self.indexed_string.word(x[0]))
        #   print(self.indexed_string.string_position(x[0]))
        #   print(x[1])
          exp_lst.append((self.indexed_string.word(x[0]), self.indexed_string.string_position(x[0]), x[1]))
        # print("======")
        # exp = [(self.indexed_string.word(x[0]),
        #         self.indexed_string.string_position(x[0]),
        #         x[1]) for x in exp]
        exp = exp_lst
        all_occurrences = list(itertools.chain.from_iterable(
            [itertools.product([x[0]], x[1], [x[2]]) for x in exp]))
        all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]

        ret = '''
            %s.show_raw_text(%s, %d, %s, %s, %s);
            ''' % (exp_object_name, json.dumps(all_occurrences), label,
                   json.dumps(text), div_name, json.dumps(opacity))
        return ret

def id_generator(size=15, random_state=None):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(random_state.choice(chars, size, replace=True))

def lime_style_visualization(text, pipeline, class_names, contributions, method_name, base_path):
  def jsonize(x):
      return json.dumps(x, ensure_ascii=False)

  from lime.lime_text import IndexedString
  indexed_string = IndexedString(text, bow=True,split_expression=r'\W+', mask_string=None)
  # print("indexed_string", indexed_string)
  domain_mapper = TextDomainMapper(indexed_string)
  # ans = domain_mapper.map_exp_ids(contributions)
  # ans = [(x[0], float(x[1])) for x in ans]
  # print("ans", ans)
  random_state = 50
  labels = [1]
  prediction = pipeline.predict([text])[0]

  bundle = open(os.path.join("all_results", 'bundle.js'), encoding="utf8").read()
  out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
  random_id = id_generator(size=15, random_state=check_random_state(random_state))
  out += u'''
  <div class="lime top_div" id="top_div%s"></div>
  ''' % random_id
  predict_value_js = ''
  predict_proba_js = u'''
  var main_div = top_div.append('div').classed('lime top_div', true);
  
  var explanation_method_name = main_div.append('h1').style('width', '100%%');

  explanation_method_name[0][0].innerHTML = '%s';


  var pp_div = main_div.append('div')
                      .classed('lime predict_proba', true);
  var pp_svg = pp_div.append('svg').style('width', '100%%');
  var pp = new lime.PredictProba(pp_svg, %s, %s);
  ''' % (method_name, jsonize([str(x) for x in class_names]),
          jsonize(list(prediction.astype(float))))
  
  exp_js = '''var exp_div;
            var exp = new lime.Explanation(%s);
        ''' % (jsonize([str(x) for x in class_names]))
  html_data = []
  exp_lst = []
  data = []
  word_to_id_dic = {}

  tmp_lst = []
  for label in labels:
      # exp = jsonize(self.as_list(label))
      for idx, word in enumerate(text.split(" ")):
        import re
        a = re.finditer(word, text)
        lst = []
        for m in a:
          lst.append(m.start(0))
        word_to_id_dic[word] = lst
        exp_lst.append((word, contributions[idx].astype(float)))
        tmp_lst.append((idx, word, contributions[idx].astype(float)))
  lst = []
  unique_words = {}
  idx = 0
  for value in exp_lst:
    # print("value",  value)
    if not value[0].lower() in unique_words.keys():
    #   print(value[0].lower(), idx)
      unique_words[value[0].lower()] = idx
      lst.append(value)#
      idx += 1
    #   print("unique_words: ", unique_words)
  exp_lst = lst


  
  for idx, tup in enumerate(exp_lst):
    html_data.append((unique_words[tup[0].lower()], tup[1]))
    # data.append((idx, word_to_id_dic[word], contributions[tup[0]].astype(float)))
  
  # exp_lst = list(set(exp_lst))
  # for idx, tup in enumerate(exp_lst):
  #   print(idx, tup)


  html_data = list(set(html_data))
  
  exp_lst = sorted(exp_lst, key=lambda tup: np.abs(tup[1]), reverse=True)
  html_data = sorted(html_data, key=lambda tup: np.abs(tup[1]), reverse=True)
  
  exp_lst = jsonize(exp_lst[:10])
  exp_js += u'''
                exp_div = main_div.append('div').classed('lime explanation', true);
                exp.show(%s, %d, exp_div);
                ''' % (exp_lst, label)

  raw_js = '''var raw_div = main_div.append('div');'''
  raw_js += domain_mapper.visualize_instance_html(
                html_data,
                labels[0],
                'raw_div',
                'exp')


  out += u'''
        <script>
        var top_div = d3.select('#top_div%s').classed('lime top_div', true);
        %s
        %s
        %s
        %s
        </script>
        ''' % (random_id, predict_proba_js, predict_value_js, exp_js, raw_js)
  out += u'</body></html>'
  
  return out


def show_results(text_seq, text_seqs, tokenizer, model, class_names, base_path, exp_contributions, activation_name, SHAP_explainer=None, 
            is_show=False, methods = ["ig", "lime", "shap", "sg"], max_len=100):
    word_index = tokenizer.word_index
    reverse_index = {value: key for (key, value) in word_index.items()}
    text = decode_sentence(text_seq[0], reverse_index)

    text_seq = text_seq.reshape(-1, max_len)
    predictions = model(text_seq)
    label = predictions.numpy().argmax(axis=1)
    pred_dict = {i: class_names[i] for i in range(len(class_names))}

    vectorizer = Vectorizer(tokenizer, max_len)
    pipeline = make_pipeline(vectorizer, model)
    
    lime_contribution = exp_contributions[0]
    shape_contribution = exp_contributions[1]
    ig_contributions = exp_contributions[2]
    sg_contributions = exp_contributions[3]
    if "ig" in methods:
        if is_show:
            # print("\n\nIntegrated Gradients")
            label = "IG :" + activation_name
            out = lime_style_visualization(text, pipeline, class_names, ig_contributions, label, base_path)
            display(HTML(out))
        words = text.split()
        # ig_contributions = explain_integrated_gradients(words, text_seq, model, is_show=is_show)
    
    if "lime" in methods:
        if is_show:
            # print("\n\nLIME")
            label = "LIME :" + activation_name
            out = lime_style_visualization(text, pipeline, class_names, lime_contribution, label, base_path)
            display(HTML(out))
        # lime_words, lime_contribution = explain_lime(text, tokenizer, model, class_names, is_show=is_show)
        # if is_show:    
        #     print("LIME contributions", lime_contribution.shape)

    if "shap" in methods:
        if is_show:
            # print("\n\nSHAP")
            label = "SHAP :" + activation_name
            out = lime_style_visualization(text, pipeline, class_names, shape_contribution, label, base_path)
            display(HTML(out))
        words = text.split()
        # shape_contribution = explain_shap(words, text_seq, label, SHAP_explainer, is_show=is_show)

    if "sg" in methods:
        # if is_show:
        #     print("\n\nSmooth Gradients")
        # sg_contributions = get_smooth_grad(text_seq, model)
        if is_show:
            # text, pipeline, class_names, contributions, method_name
            label = "SG :" + activation_name
            out = lime_style_visualization(text, pipeline, class_names, sg_contributions, label, base_path)
            # from IPython.display import display, HTML
            display(HTML(out))
            # colors = colorize(sg_contributions)
            # display(HTML("".join(list(map(hlstr, words, colors)))))
    
    return lime_contribution, shape_contribution, ig_contributions, sg_contributions
    
    
def show_for_kaggle(path, data_name, data, seqs_count, explanation_methods, robust=False,
                    custom_dropouts=([None, 10, 20, 30, 40, 50]), 
                    max_len=100, is_show=True):
    x_tr_seq, y_train, x_val_seq, y_test, tokenizer, size_of_vocabulary = data
    activations = [None, inverse_sigmoid, hard_sigmoid, sinusoidal_sigmoid]
    activation_names = ['No Tampering', 'Inverse Sigmoid', 'Hard Sigmoid', 'Sinusoidal Sigmoid']
    class_names = ["Real", "Fake"]
    
    for custom_dropout in custom_dropouts:
        print("dropout_value: ", custom_dropout)
        # x_tr_seq[:seqs_count]
        # text_labels = np.argmax(models[0].predict(), axis=1)
        
        if robust:
            filename = "robust_hybrid_"+data_name+"_"+str(max_len)+"_"+str(custom_dropout)+".npy"
        else:
            filename = "hybrid_"+data_name+"_"+str(max_len)+"_"+str(custom_dropout)+".npy"
        # exp_contributions = np.load(path+filename)
        model = load_trained_model(path, size_of_vocabulary, data_name=data_name, 
                                num_classes=y_train.shape[1], custom_dropout=custom_dropout)
        class_names = ["Real", "Fake"]
        model_unmasked = inference_model(model, include_softmax=False)
        min_threshold, max_threshold, _ = visualize_activations(model_unmasked, x_tr_seq[:100])
        
        models = []
        accuracies = []
        for i in range(len(activations)):
            models.append(inference_model(model, tampering_activation=activations[i], 
                                        tampering_limits=([min_threshold, max_threshold])))
            accuracies.append(models[-1].evaluate(x_val_seq, y_test)[1])
        

        # 0th index for: original:0 vs attack model:1
        # 1th index for: LIME:0, SHAP:1, IG:2,  SG:3 
        # lime_contributions = np.zeros((len(text_seqs), 100))
        print(path+filename)
        exp_contributions = np.load(path+filename)
        exp_contributions = exp_contributions/np.expand_dims(np.sum(np.abs(exp_contributions), axis=3), axis=3)
        # n=3
        # exp_contributions = (exp_contributions * 10**n).astype(np.int32)
        # exp_contributions = exp_contributions.astype(np.float32)/(10**n)
        
        for n in seqs_count:
            for i, modell in enumerate(models):
                cons = np.zeros((4, 100))
                cons[0] = exp_contributions[i, 0, n, :]
                cons[1] = exp_contributions[i, 1, n, :]
                cons[2] = exp_contributions[i, 2, n, :]
                cons[3] = exp_contributions[i, 3, n, :]

                show_results(
                            text_seq=x_tr_seq[n:n+1], 
                            text_seqs=x_tr_seq[n:n+1], 
                            tokenizer=tokenizer,
                            model=modell, 
                            class_names=class_names, 
                            base_path=path,
                            is_show=is_show, 
                            methods = explanation_methods,
                            exp_contributions =cons,
                            activation_name= activation_names[i]
                          )
                
                # print("custom_dropout = ", custom_dropout, "i=", i, " n = ", n, " time =", time.time()-tik)
    return exp_contributions