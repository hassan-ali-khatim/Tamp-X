import numpy as np
import time
import math
from general_utils import load_tokenized_data, make_custom_ag_data, visualize_activations, load_trained_model, get_cosine_similarity, confirm_directory
from tampering_utils import inference_model, sinusoidal_sigmoid, inverse_sigmoid, hard_sigmoid, shap, plot_activation_function
from xai_utils import explain
import matplotlib.pyplot as plt

SMALL_SIZE = 17
MEDIUM_SIZE = 21
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def accuracies_for_different_settings(data, data_name, verbose=False, path='reults/',
                                      custom_dropouts=([None, 10, 20, 30, 40, 50])):
    x_tr_seq, y_train, x_val_seq, y_test, tokenizer, size_of_vocabulary = data
    activations = [inverse_sigmoid, hard_sigmoid, sinusoidal_sigmoid]
    
    for custom_dropout in custom_dropouts:
        model = load_trained_model(path, size_of_vocabulary, data_name=data_name, 
                                num_classes=y_train.shape[1], custom_dropout=custom_dropout)
        print("Accuracy with the z-masking layer", 
            model.evaluate(x_val_seq, y_test, batch_size=64, verbose=False)[1])
        model_unmasked = inference_model(model, include_softmax=True)
        print("Accuracy without the z-masking layer", 
            model_unmasked.evaluate(x_val_seq, y_test, batch_size=64, verbose=False)[1])
        model_unmasked2 = inference_model(model, include_softmax=False)
        min_threshold, max_threshold, _ = visualize_activations(model_unmasked2, x_tr_seq[:100])
        for activation in activations:
            model_tampered = inference_model(model, tampering_activation=activation, 
                                            tampering_limits=[min_threshold, max_threshold],
                                            include_softmax=True, binary_class=None)
            print("Accuracy after tampering", 
                model_tampered.evaluate(x_val_seq, y_test, batch_size=64, verbose=False)[1])
            if verbose:
                plot_activation_function(activation(limits=[min_threshold, max_threshold]))


def compute_accuracies(data, data_name="kaggle", custom_dropouts=([None, 10, 20, 30, 40, 50]), 
                    max_len=100, path="results/"):
    
    x_tr_seq, y_train, x_val_seq, y_test, tokenizer, size_of_vocabulary = data
    activations = [None, inverse_sigmoid, hard_sigmoid, sinusoidal_sigmoid]
    activation_names = ["No Tampering", "Inverse Sigmoid", "Hard Sigmoid", "Sinusoidal Sigmoid"]
    
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    accuracies_all = []
    for cd, custom_dropout in enumerate(custom_dropouts):
        print("dropout_value: ", custom_dropout)
        model = load_trained_model(path, size_of_vocabulary, data_name=data_name, 
                                num_classes=y_train.shape[1], custom_dropout=custom_dropout)
        model_unmasked = inference_model(model, include_softmax=False)
        min_threshold, max_threshold, _ = visualize_activations(model_unmasked, x_tr_seq[:100])
            
        accuracies = []
        for i in range(len(activations)):
            model_tampered = inference_model(model, tampering_activation=activations[i], 
                                    tampering_limits=([min_threshold, max_threshold]))
            accuracies.append(model_tampered.evaluate(x_val_seq, y_test)[1])
        accuracies_all.append(accuracies)
    accuracies_all = np.array(accuracies_all)
    np.save(path+"accuracies_"+data_name+"_"+str(max_len)+".npy", accuracies_all)
    return
    
    
def plot_accuracies(data_name, custom_dropouts=([None, 10, 20, 30, 40, 50]), 
                    max_len=100, path="results/"):

    width = 0.1
    colors = ["coral", "skyblue", "lawngreen", "cornflowerblue", "limegreen", "lavender", "slateblue", "mediumorchid"]
    activations = [None, inverse_sigmoid, hard_sigmoid, sinusoidal_sigmoid]
    activation_names = ["No Tampering", "Inverse Sigmoid", "Hard Sigmoid", "Sinusoidal Sigmoid"]
    
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    
    br = np.arange(len(custom_dropouts))
    br1 = [i+width*1.4 for i in br]
    br2 = [i+width*2.8 for i in br]
    br3 = [i+width*4.2 for i in br]

    brs = [br, br1, br2, br3]
    
    custom_dropouts2 = []
    for custom_dropout in custom_dropouts:
        if custom_dropout is None:
            custom_dropouts2.append(0)
        else:
            custom_dropouts2.append(custom_dropout)
    
    accuracies_all = np.load(path+"accuracies_"+data_name+"_"+str(max_len)+".npy")
    for act_index, activation in enumerate(activations):
        ax.bar(brs[act_index], accuracies_all[:, act_index], width, 
               edgecolor='black', color=colors[act_index], label=activation_names[act_index])# if cd==0 else None)
    
    ax.set_xticks(np.array(br1))
    # ax.set_ylim(-1, 1)
    ax.set_xticklabels(custom_dropouts2)
    ax.set_ylabel("Accuracies")
    ax.set_xlabel("z")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=2, edgecolor="black", fancybox=False)
    plt.tight_layout()
    return fig


def explain_for_kaggle(data, seqs_count, custom_dropouts=([None, 10, 20, 30, 40, 50]), 
                       max_len=100, path="results/"):
    data_name = "kaggle"
    x_tr_seq, y_train, x_val_seq, y_test, tokenizer, size_of_vocabulary = data
    is_show =False # show explainations?
    activations = [None, inverse_sigmoid, hard_sigmoid, sinusoidal_sigmoid]
    constant_2s = [5, 5, 5, 5]
    class_names = ["Positive", "Negative"]
    
    for custom_dropout in custom_dropouts:
        print("dropout_value: ", custom_dropout)
        model = load_trained_model(path, size_of_vocabulary, data_name=data_name, 
                                num_classes=y_train.shape[1], custom_dropout=custom_dropout)
        class_names = ["Negative", "Positive"]
        model_unmasked = inference_model(model, include_softmax=False)
        min_threshold, max_threshold, _ = visualize_activations(model_unmasked, x_tr_seq[:100])
            
        models = []
        accuracies = []
        for i in range(len(activations)):
            models.append(inference_model(model, tampering_activation=activations[i], 
                                        tampering_limits=([min_threshold, max_threshold])))
            accuracies.append(models[-1].evaluate(x_val_seq, y_test)[1])
        
        text_labels = np.argmax(models[0].predict(x_tr_seq[:seqs_count]), axis=1)
        exp_contributions = np.zeros((len(models), 4, seqs_count, max_len))
        
        # 0th index for: original:0 vs attack model:1
        # 1th index for: LIME:0, SHAP:1, IG:2,  SG:3 
        # lime_contributions = np.zeros((len(text_seqs), 100))
        for i, modell in enumerate(models):
            SHAP_explainer = shap.KernelExplainer(modell.predict, x_tr_seq[:seqs_count])
            for n in range(seqs_count):
                tik = time.time()
                cons = explain(x_tr_seq[n:n+1], x_tr_seq[:seqs_count], text_labels[n], tokenizer, modell, 
                            class_names, SHAP_explainer=SHAP_explainer, is_show=is_show, 
                            methods = ["lime", "shap", "ig", "sg"])
                
                print("custom_dropout = ", custom_dropout, "i=", i, " n = ", n, " time =", time.time()-tik)
                print(cons[0].shape)
                exp_contributions[i, 0, n, :] = cons[0]
                exp_contributions[i, 1, n, :] = cons[1]
                exp_contributions[i, 2, n, :] = cons[2]
                exp_contributions[i, 3, n, :] = cons[3]

        np.save(path+"hybrid_"+data_name+"_"+str(max_len)+"_"+str(custom_dropout)+".npy", np.nan_to_num(exp_contributions))


def explain_for_ag(data, seqs_count, custom_dropouts=([None, 10, 20, 30, 40, 50]), 
                   max_len=100, path="results/"):
    data_name = "AG"
    x_tr_seq, y_train, x_val_seq, y_test, tokenizer, size_of_vocabulary = data
    is_show =False # show explainations?
    activations = [None, inverse_sigmoid, hard_sigmoid, sinusoidal_sigmoid]
    constant_2s = [5, 5, 5, 5]
    class_names = ["Positive", "Negative"]
    
    for custom_dropout in custom_dropouts:
        accuracies_all = []
        model = load_trained_model(path, size_of_vocabulary, data_name=data_name, 
                                   num_classes=y_train.shape[1], custom_dropout=custom_dropout)
        for binary_class in range(y_train.shape[1]):
            models = []
            accuracies = []
            model_unmasked2 = inference_model(model, include_softmax=False)
            min_threshold, max_threshold, _ = visualize_activations(model_unmasked2, x_tr_seq[:100])
            
            for i in range(len(activations)):
                model_tampered = inference_model(model, tampering_activation=activations[i], 
                                                tampering_limits=[min_threshold, max_threshold], 
                                                include_softmax=True, binary_class=None)
                accuracies.append(model_tampered.evaluate(x_val_seq, y_test)[1])
                model_tampered = inference_model(model, tampering_activation=activations[i], 
                                                tampering_limits=[min_threshold, max_threshold], 
                                                include_softmax=True, binary_class=binary_class)
                models.append(model_tampered)
            accuracies_all.append(accuracies)

            # 0th index for: original:0 vs attack model:1
            # 1th index for: LIME:0, SHAP:1, IG:2,  SG:3 
            # lime_contributions = np.zeros((len(text_seqs), 100))
            x_xai = x_tr_seq[np.where(np.argmax(y_train, axis=1)==binary_class)][:seqs_count]
            text_labels = np.argmax(models[0].predict(x_xai[:seqs_count]), axis=1)
            exp_contributions = np.zeros((len(models), 4, seqs_count, max_len))
            for i, modell in enumerate(models):
                SHAP_explainer = shap.KernelExplainer(modell.predict, x_xai[:seqs_count])
                for n in range(seqs_count):
                    tik = time.time()
                    cons = explain(x_xai[n:n+1], x_xai[:seqs_count], text_labels[n], tokenizer, modell, 
                                class_names, SHAP_explainer=SHAP_explainer, is_show=is_show, 
                                methods = ["lime", "shap", "ig", "sg"])

                    print("custom_dropout = ", custom_dropout, "i=", i, " n = ", n, " time =", time.time()-tik)
                    print(cons[0].shape)
                    exp_contributions[i, 0, n, :] = cons[0]
                    exp_contributions[i, 1, n, :] = cons[1]
                    exp_contributions[i, 2, n, :] = cons[2]
                    exp_contributions[i, 3, n, :] = cons[3]
            
            # exp_contributions_all.append(exp_contributions)
            if binary_class == 0:
                exp_contributions_all = exp_contributions
            else:
                exp_contributions_all = np.append(exp_contributions_all, exp_contributions, axis=2)
        
        print("The cumulative contributions shape: ", exp_contributions_all.shape)
        np.save(path+"hybrid_"+data_name+"_"+str(max_len)+"_"+str(custom_dropout)+".npy", 
                np.nan_to_num(exp_contributions_all))
        
        
def explain_for_toxic(data, seqs_count, custom_dropouts=([None, 10, 20, 30, 40, 50]), 
                   max_len=100, path="results/"):
    data_name = "toxic"
    x_tr_seq, y_train, x_val_seq, y_test, tokenizer, size_of_vocabulary = data
    is_show =False # show explainations?
    activations = [None, inverse_sigmoid, hard_sigmoid, sinusoidal_sigmoid]
    constant_2s = [5, 5, 5, 5]
    class_names = ["Positive", "Negative"]
    
    for custom_dropout in custom_dropouts:
        accuracies_all = []
        model = load_trained_model(path, size_of_vocabulary, data_name=data_name, 
                                   num_classes=y_train.shape[1], custom_dropout=custom_dropout)
        for binary_class in range(y_train.shape[1]):
            models = []
            accuracies = []
            model_unmasked2 = inference_model(model, include_softmax=False)
            min_threshold, max_threshold, _ = visualize_activations(model_unmasked2, x_tr_seq[:100])
                
            for i in range(len(activations)):
                model_tampered = inference_model(model, tampering_activation=activations[i], 
                                                tampering_limits=[min_threshold, max_threshold], 
                                                include_softmax=True, binary_class=None)
                accuracies.append(model_tampered.evaluate(x_val_seq, y_test)[1])
                model_tampered = inference_model(model, tampering_activation=activations[i], 
                                                tampering_limits=[min_threshold, max_threshold], 
                                                include_softmax=True, binary_class=binary_class)
                models.append(model_tampered)
            accuracies_all.append(accuracies)

            # 0th index for: original:0 vs attack model:1
            # 1th index for: LIME:0, SHAP:1, IG:2,  SG:3 
            # lime_contributions = np.zeros((len(text_seqs), 100))
            x_xai = x_tr_seq[np.where(np.argmax(y_train, axis=1)==binary_class)][:seqs_count]
            exp_contributions = np.zeros((len(models), 4, seqs_count, max_len))
            if len(x_xai) > 0:
                text_labels = np.argmax(models[0].predict(x_xai[:seqs_count]), axis=1)
                for i, modell in enumerate(models):
                    SHAP_explainer = shap.KernelExplainer(modell.predict, x_xai[:seqs_count])
                    for n in range(len(x_xai)):
                        tik = time.time()
                        cons = explain(x_xai[n:n+1], x_xai[:seqs_count], text_labels[n], tokenizer, modell, 
                                    class_names, SHAP_explainer=SHAP_explainer, is_show=is_show, 
                                    methods = ["lime", "shap", "ig", "sg"])

                        print("custom_dropout = ", custom_dropout, "i=", i, " n = ", n, " time =", time.time()-tik)
                        print(cons[0].shape)
                        exp_contributions[i, 0, n, :] = cons[0]
                        exp_contributions[i, 1, n, :] = cons[1]
                        exp_contributions[i, 2, n, :] = cons[2]
                        exp_contributions[i, 3, n, :] = cons[3]
                
            if binary_class == 0:
                exp_contributions_all = exp_contributions
            else:
                exp_contributions_all = np.append(exp_contributions_all, exp_contributions, axis=2)
            np.save(path+"hybrid_"+data_name+"_"+str(max_len)+"_"+str(custom_dropout)+".npy", 
                    np.nan_to_num(exp_contributions_all))
        
        print("The cumulative contributions shape: ", exp_contributions_all.shape)
        np.save(path+"hybrid_"+data_name+"_"+str(max_len)+"_"+str(custom_dropout)+".npy", 
                np.nan_to_num(exp_contributions_all))


def plot_L_Norms(norm_used, data_name="kaggle", seqs_count=10, path="results/", 
                custom_dropouts=([None, 10, 20, 30, 40, 50]), max_len=100):
    norm_used = str(norm_used)
    width = 0.1
    colors = ["coral", "skyblue", "lawngreen", "cornflowerblue", "limegreen", "lavender", "slateblue", "mediumorchid"]
    activation_layers = [inverse_sigmoid, hard_sigmoid, sinusoidal_sigmoid]
    activation_layers_names = ["Inverse Sigmoid", "Hard Sigmoid", "Sinusoidal Sigmoid"]
    X_methods = ["LIME", "SHAP", "IG", "SG"]
    scores = np.zeros((len(custom_dropouts), len(activation_layers), len(X_methods), seqs_count))

    br = np.arange(len(X_methods))
    br1 = [i+width*1.4 for i in br]
    br2 = [i+width*2.8 for i in br]
    br3 = [i+width*2.8 for i in br]

    brs = [br, br1, br2, br3]
    figs = []

    for cd, custom_dropout in enumerate(custom_dropouts):
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        filename = path + "hybrid_"+data_name+"_"+str(max_len)+"_"+str(custom_dropout)+".npy"
        contributions = np.load(filename)
        contributions = np.nan_to_num(contributions)
        for act_index, act_layer in enumerate(activation_layers_names):
            for n_method, X_method in enumerate(X_methods):     
                for sample_no in range(seqs_count):
                    st_con = contributions[0, n_method, sample_no, :]
                    nd_con = contributions[act_index+1, n_method, sample_no, :]

                    def normalize_norm(arr_in):
                        if np.sum(arr_in) == 0:
                            arr_in = arr_in + 1e-11
                        arr_in_mag = np.sqrt(np.sum(arr_in**2))
                        return arr_in/arr_in_mag, arr_in_mag

                st_con_norm, s_norm = normalize_norm(st_con)
                nd_con_norm, n_norm = normalize_norm(nd_con)

                l2_norm = np.sqrt(np.sum((st_con_norm - nd_con_norm)**2))
                l1_norm = np.sum(np.abs(st_con_norm - nd_con_norm))
                l_norms = {'2': l2_norm, '1': l1_norm}
                if (normalize_norm(n_norm)[1] - 1)>0.2:
                      print("this is alarming !")
                scores[cd, act_index, n_method, sample_no] = l_norms[norm_used]

            ax.bar(brs[act_index], np.mean(scores[cd,act_index],axis=1), width, 
                                  edgecolor='black', color=colors[act_index], label=act_layer)# if cd==0 else None)
        ax.set_xticks(np.array(br1))
        # ax.set_ylim(-1, 1)
        ax.set_xticklabels(X_methods)
        ax.set_ylabel("$L_"+norm_used+"$ Norm")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, edgecolor="black", fancybox=False)
        plt.tight_layout()
        figs.append(fig)
        
    import pandas as pd
    df = pd.DataFrame()
    df["z_values"] = custom_dropouts
    for act_index, act_layer in enumerate(activation_layers_names):
        for n_method, X_method in enumerate(X_methods):
            df[act_layer+"_"+X_method] = np.mean(scores[:, act_index, n_method], axis=1)
    confirm_directory(path+'csvs/')
    df.to_csv(path+"csvs/hybrid_"+data_name+"_"+str(max_len)+'L'+norm_used+'_norms.csv', encoding='utf-8', index=False)
    return figs


def get_accuracy(topn, text_seqs, text_labels, model, con, is_remove=True):
    b_topn_indices = np.array(np.argsort(-1 * con)[:topn])
    if is_remove:
        seqs = np.copy(text_seqs)
        seqs[0, b_topn_indices] = 0
    else:
        seqs = np.zeros_like(text_seqs)
        seqs[0, b_topn_indices] = text_seqs[0, b_topn_indices]
    text_predictions = np.argmax(model.predict(seqs), axis=1)#np.argmax(y_test[54])
    # acc = np.sum(text_predictions == text_labels)/len(text_seqs)
    return (text_predictions[0] == np.argmax(text_labels,axis=1)[0])


def plot_descriptive_accuracies(data, data_name="kaggle", seqs_count=10, path="results/", 
                                custom_dropouts=([None, 10, 20, 30, 40, 50]), max_len=100,
                                num_words=10, is_remove=True):
    width = 0.1
    colors = ["coral", "skyblue", "lawngreen", "cornflowerblue", "limegreen", "lavender", "slateblue", "mediumorchid"]
    activation_layers = [None, inverse_sigmoid, hard_sigmoid, sinusoidal_sigmoid]
    activation_layers_names = ["No Tampering", "Inverse Sigmoid", "Hard Sigmoid", "Sinusoidal Sigmoid"]
    X_methods = ["LIME", "SHAP", "IG", "SG"]
    scores = np.zeros((len(custom_dropouts), len(activation_layers_names), len(X_methods), seqs_count))

    br = np.arange(len(X_methods))
    br1 = [i+width*1.4 for i in br]
    br2 = [i+width*2.8 for i in br]
    br3 = [i+width*4.2 for i in br]
    br4 = [i+width*5.6 for i in br]

    brs = [br, br1, br2, br3, br4]
    figs = []

    for cd, custom_dropout in enumerate(custom_dropouts):
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        filename = path + "hybrid_"+data_name+"_"+str(max_len)+"_"+str(custom_dropout)+".npy"
        contributions = np.load(filename)
        contributions = np.nan_to_num(contributions)
        x_tr_seq, y_train, x_val_seq, y_test, tokenizer, size_of_vocabulary = data
        model_a = load_trained_model(path, size_of_vocabulary, data_name=data_name, 
                                         num_classes=y_train.shape[1], custom_dropout=custom_dropout)
        model_unmasked = inference_model(model_a, include_softmax=False)
        min_threshold, max_threshold, _ = visualize_activations(model_unmasked, x_tr_seq[:100])

        for act_index, act_layer in enumerate(activation_layers_names):  
            model = inference_model(model_a, tampering_activation=activation_layers[act_index],
                                    tampering_limits=[min_threshold, max_threshold])
            for n_method, X_method in enumerate(X_methods):     
                for sample_no in range(seqs_count):
                    nd_con = contributions[act_index, n_method, sample_no, :]
                    cosine_sim = get_accuracy(num_words, x_tr_seq[sample_no:sample_no+1], y_train[sample_no:sample_no+1], 
                                              model, nd_con, is_remove=is_remove)
                    scores[cd, act_index, n_method, sample_no] = cosine_sim

            if is_remove:
                ax.bar(brs[act_index], 1-np.mean(scores[cd,act_index],axis=1), width, 
                                edgecolor='black', color=colors[act_index], label=act_layer)# if cd==0 else None)
            else:
                ax.bar(brs[act_index], np.mean(scores[cd,act_index],axis=1), width, 
                                edgecolor='black', color=colors[act_index], label=act_layer)# if cd==0 else None)
        ax.set_xticks(np.array(br1))
        ax.set_xticklabels(X_methods)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, edgecolor="black", fancybox=False)
        ax.set_ylabel("Descriptive Accuracy")
        plt.tight_layout()
        figs.append(fig)
        
        np.save(path+"hybrid_"+data_name+"_descriptive_accuracies_"+str(is_remove)+".npy", scores)
        
    import pandas as pd
    df = pd.DataFrame()
    df["z_values"] = custom_dropouts
    for act_index, act_layer in enumerate(activation_layers_names):
        for n_method, X_method in enumerate(X_methods):
            df[act_layer+"_"+X_method] = np.mean(scores[:, act_index, n_method], axis=1)
    confirm_directory(path+'csvs/')
    df.to_csv(path+"csvs/hybrid_"+data_name+"_"+str(max_len)+'descriptive_accuracy'+
              str(num_words)+"_"+str(is_remove)+'.csv', encoding='utf-8', index=False)
    return figs


def plot_cosine_similarities(data_name="kaggle", seqs_count=10, path="results/", 
                             custom_dropouts=([None, 10, 20, 30, 40, 50]), max_len=100):
    width = 0.1
    colors = ["coral", "skyblue", "lawngreen", "cornflowerblue", "limegreen", "lavender", "slateblue", "mediumorchid"]
    activation_layers = [inverse_sigmoid, hard_sigmoid, sinusoidal_sigmoid]
    activation_layers_names = ["Inverse Sigmoid", "Hard Sigmoid", "Sinusoidal Sigmoid"]
    X_methods = ["LIME", "SHAP", "IG", "SG"]
    scores = np.zeros((len(custom_dropouts), len(activation_layers), len(X_methods), seqs_count))

    br = np.arange(len(X_methods))
    br1 = [i+width*1.4 for i in br]
    br2 = [i+width*2.8 for i in br]
    br3 = [i+width*2.8 for i in br]

    brs = [br, br1, br2, br3]
    figs = []

    for cd, custom_dropout in enumerate(custom_dropouts):
        fig, ax = plt.subplots(1, 1, figsize=(10,5))
        filename = path + "hybrid_"+data_name+"_"+str(max_len)+"_"+str(custom_dropout)+".npy"
        contributions = np.load(filename)
        contributions = np.nan_to_num(contributions)

        for act_index, act_layer in enumerate(activation_layers_names):
            for n_method, X_method in enumerate(X_methods):     
                for sample_no in range(seqs_count):
                    st_con = contributions[0, n_method, sample_no, :]
                    nd_con = contributions[act_index+1, n_method, sample_no, :]
                    cosine_sim = get_cosine_similarity(st_con, nd_con)
                    if math.isnan(cosine_sim):
                        cosine_sim = 0
                    scores[cd, act_index, n_method, sample_no] = cosine_sim

            ax.bar(brs[act_index], np.mean(scores[cd,act_index],axis=1), width, 
                                    edgecolor='black', color=colors[act_index], label=act_layer)
            ax.set_xticks(np.array(br1))
            # ax.set_ylim(-1, 1)
            ax.set_xticklabels(X_methods)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3, edgecolor="black", fancybox=False)
            ax.set_ylabel("Cosine Similarity")
        plt.tight_layout()
        figs.append(fig)
        
    import pandas as pd
    df = pd.DataFrame()
    df["z_values"] = custom_dropouts
    for act_index, act_layer in enumerate(activation_layers_names):
        for n_method, X_method in enumerate(X_methods):
            df[act_layer+"_"+X_method] = np.mean(scores[:, act_index, n_method], axis=1)
    confirm_directory(path+'csvs/')
    df.to_csv(path+"csvs/hybrid_"+data_name+"_"+str(max_len)+'cosine_similarity.csv', encoding='utf-8', index=False)
    return figs, scores
        

def main():
    path_data = "data"
    data_names = ["AG"]
    max_len = 100
    custom_dropouts = [None, 10, 20, 30, 40, 50]
    batch_size = 64

    explain_for_dataset = {
        "kaggle": explain_for_kaggle,
        "AG": explain_for_ag,
        "toxic": explain_for_toxic
    }
    
    for data_name in data_names:
        data = load_tokenized_data(batch_size=batch_size, data_name=data_name)
        explain_for_dataset[data_name](data, 3, path="results2/")