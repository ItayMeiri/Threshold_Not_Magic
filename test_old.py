import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import GOOD as g
import my_utils
import torch

lines = [
    (0, (1, 10)),
    (0, (1, 1)),
    (0, (1, 1)),
    (0, (5, 10)),
    (0, (5, 5)),
    (0, (5, 1)),

    (0, (3, 10, 1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (3, 1, 1, 1)),

    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 10, 1, 10, 1, 10)),
    (0, (3, 1, 1, 1, 1, 1))]

display_labels_malware = np.array(["Benign", "Malware", "OOD"])
display_labels_applications = np.array(
    ["dropbox", "facebook", "google", "microsoft", "teamviewer", "twitter", "youtube", "uncategorized", "OOD"])


def load_app():
    data, labels = my_utils.load_known_applications()
    data_u, labels_u = my_utils.load_unknown_applications(training=False)
    # insert training from both sets
    _, data, _, labels = train_test_split(data, labels, test_size=0.20, random_state=42)
    # _, data_u, _, labels_u = train_test_split(data_u, labels_u, test_size=0.5, random_state=42)  # 0.5 train 0.5 test

    labels_u[labels_u > -1] = np.max(labels) + 1  # should be all 8s
    data = np.concatenate([data, data_u])
    labels = np.concatenate([labels, labels_u])

    return data, labels


def load_malware():
    data, labels = my_utils.load_known_malware()
    data_u, labels_u = my_utils.load_unknown_malware(K1=True, training=False)
    # insert training from both sets
    _, data, _, labels = train_test_split(data, labels, test_size=0.20, random_state=42)
    _, data_u, _, labels_u = train_test_split(data_u, labels_u, test_size=0.5, random_state=42)  # 0.5 train 0.5 test

    labels_u[labels_u > -1] = np.max(labels) + 1  # should be all 2s
    data = np.concatenate([data, data_u])
    labels = np.concatenate([labels, labels_u])

    return data, labels


def split(x, y):
    return train_test_split(x, y, test_size=0.25, random_state=42)


def plot_odin(max_predictions, argmax_predictions, labels, og_argmax_predictions, title="Malware", save_name="title"):
    mx_label = np.max(labels)
    acc, recall, tn, tp, reject, scores, ood_r = [], [], [], [], [], [], []
    x = []

    best_a = -1
    best_t = -1

    percent = 0

    for thresh in np.arange(0, 100, 0.01):
        thresh = np.percentile(max_predictions, thresh)

        if thresh % 0.01 == 0:
            print("percent complete:", percent)
            percent += 1
        ood_idx = max_predictions < thresh
        new_predictions = np.copy(og_argmax_predictions)
        new_predictions[ood_idx] = mx_label

        d = my_utils.evaluate_without_model(new_predictions, labels, show_matrix=False, samples=True, use_torch=False,
                                            excluded=True)
        x.append(thresh)
        if d[0] > best_a and d[4] < 0.5:
            best_a = d[0]
            best_t = thresh
        acc.append(d[0])
        recall.append(d[1])
        tn.append(d[2])
        tp.append(d[3])
        reject.append(d[4])

        # get statistics
    print("best a", best_a, "best t", best_t)
    plt.xlabel("Threshold")
    plt.ylabel("Percentage")
    plt.title("ODIN - " + title)
    plt.plot(x, acc, label="accuracy", linestyle='dashed')
    plt.plot(x, recall, label="recall", linestyle=lines[5])
    plt.plot(x, tn, label="true negatives (IND detected as IND", linestyle=lines[1])
    plt.plot(x, tp, label="true positives(OOD detected as OOD)", linestyle=lines[3])
    plt.plot(x, reject, label="samples rejected%", linestyle=lines[9])
    # plt.plot(x, ood_r, label="OOD recall", linestyle=lines[11])
    # plt.plot(x, scores, label= "scores")
    plt.rcParams.update({'font.size': 16})

    plt.legend(loc='best')

    plt.savefig("images/" + save_name,bbox_inches='tight')
    plt.clf()
    # plt.show()
    return best_a, best_t


def plot_bp(r, labels, grads, title="Malware", save_name="title"):
    mx = torch.max(labels)
    acc, recall, tn, tp, reject, scores, ood_r = [], [], [], [], [], [], []
    x = []

    best_a = -1
    best_t = -1
    for t in np.arange(0, 100, 0.1):
        t = np.percentile(grads, t)
        copy_r = r.detach().clone()
        argmax_preds = torch.argmax(copy_r, axis=-1)
        argmax_preds[grads > t] = mx

        d = my_utils.evaluate_without_model(argmax_preds, labels, show_matrix=False, samples=True, excluded=True)

        if d[0] > best_a and d[4] < 0.5:
            best_a = d[0]
            best_t = t

        x.append(t)
        acc.append(d[0])
        recall.append(d[1])
        tn.append(d[2])
        tp.append(d[3])
        reject.append(d[4])

    # scores = np.array(scores) / np.linalg.norm(np.array(scores))
    # x.reverse()
    # plt.gca().invert_xaxis()
    plt.xlabel("Threshold")
    plt.ylabel("Percentage")
    plt.title("Backpropagation - " + title)
    plt.plot(x, acc, label="accuracy", linestyle='dashed')
    plt.plot(x, recall, label="recall", linestyle=lines[5])
    plt.plot(x, tn, label="true negatives (IND detected as IND", linestyle=lines[1])
    plt.plot(x, tp, label="true positives(OOD detected as OOD)", linestyle=lines[3])
    plt.plot(x, reject, label="samples rejected%", linestyle=lines[9])
    # plt.plot(x, ood_r, label="OOD recall", linestyle=lines[11])
    plt.rcParams.update({'font.size': 16})
    plt.legend(loc='best')
    plt.savefig("images/" + save_name,bbox_inches='tight')
    # plt.show  ()
    plt.clf()

    return best_a, best_t


def test_bp_malware(data, labels):
    print("TEST BP MALWARE - BEGIN")
    # data, labels = my_utils.load_known_malware()
    # data_u, labels_u = my_utils.load_unknown_malware()
    # _, data, _, labels = split(data, labels)
    # _, data_u, _, labels_u = split(data_u, labels_u)
    #
    # labels_u[labels_u > -1] = np.max(labels) + 1  # should be all 2s
    # data = np.concatenate([data, data_u])
    # labels = np.concatenate([labels, labels_u])

    # data, labels = load_malware()
    data = torch.tensor(data)
    labels = torch.tensor(labels)

    model = my_utils.get_malware_model("pytorch")
    grads, r = g.shadow_backpropagation(model, model.CNN[6], data)

    mx_d = torch.max(r) + 1

    best_a, best_t = plot_bp(r, labels, grads, title="Malware", save_name="malware_app")

    print("BEST ACCURACY BP MALWARE:", best_a)
    copy_r = r.detach().clone()
    argmax_preds = torch.argmax(copy_r, axis=-1)
    argmax_preds[grads > best_t] = torch.max(labels)

    d = my_utils.evaluate_without_model(argmax_preds, labels, show_matrix=True, samples=True,
                                        display_labels=display_labels_malware, excluded=True, title="bp_malware_cf")
    print("TEST BP MALWARE - END")


def test_bp_app(data, labels):
    print("TEST BP APP - BEGIN")

    # data, labels = my_utils.load_known_applications()
    # data_u, labels_u = my_utils.load_unknown_applications()
    # _, data, _, labels = split(data, labels)
    # labels_u[labels_u > -1] = np.max(labels) + 1
    # data = np.concatenate([data, data_u])
    # labels = np.concatenate([labels, labels_u])
    #

    data, labels = load_app()
    data = torch.tensor(data)
    labels = torch.tensor(labels)

    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    model = my_utils.get_application_model("pytorch")
    grads, r = g.shadow_backpropagation(model, model.CNN[6], data)

    # threshold = my_utils.calibrate_threshold(data, labels, grads)

    mx = torch.max(labels)
    best_a, best_t = plot_bp(r, labels, grads, title="Application", save_name="bp_app")

    print("BEST ACCURACY BP APP:", best_a)

    copy_r = r.detach().clone()
    argmax_preds = torch.argmax(copy_r, axis=-1)
    argmax_preds[grads > best_t] = mx

    d = my_utils.evaluate_without_model(argmax_preds, labels, show_matrix=True, samples=True,
                                        display_labels=display_labels_applications, excluded=True, title="bp_app_cf")

    print("TEST BP APP - END")


def test_odin_malware(data, labels):
    print("TEST ODIN MALWARE - BEGIN")
    #
    # data, labels = my_utils.load_known_malware()
    # data_u, labels_u = my_utils.load_unknown_malware()
    # _, data, _, labels = split(data, labels)
    # _, data_u, _, labels_u = split(data_u, labels_u)
    #
    # labels_u[labels_u > -1] = np.max(labels) + 1
    #
    # data = np.concatenate([data, data_u])
    # labels = np.concatenate([labels, labels_u])
    # data, labels = load_malware()
    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    model = my_utils.get_malware_model("tensorflow")

    og_argmax_predictions = np.argmax(model.predict(data), axis=-1)
    # need to search ideal eps
    p_data = my_utils.create_perturbed_ds(data, model, 0.001)

    raw_predictions = model.predict(p_data)
    argmax_predictions = np.argmax(raw_predictions, axis=-1)
    max_predictions = np.max(raw_predictions, axis=-1)

    acc, recall, tn, tp, reject, scores, ood_r = [], [], [], [], [], [], []
    x = []

    best_a, best_t = plot_odin(max_predictions, argmax_predictions, labels, og_argmax_predictions,
                               save_name="odin_malware")

    print("BEST ACCURACY ODIN MALWARE:", best_a)

    ood_idx = max_predictions < best_t
    new_predictions = np.copy(argmax_predictions)
    new_predictions[ood_idx] = np.max(labels)  # change labels to OOD when detected
    new_predictions[~ood_idx] = og_argmax_predictions[~ood_idx]

    d = my_utils.evaluate_without_model(new_predictions, labels, show_matrix=True, samples=True, use_torch=False,
                                        display_labels=display_labels_malware, excluded=True, title="odin_malware_cf")

    print("TEST ODIN MALWARE - END")


def test_odin_app(data, labels, psize=0.04):
    print("TEST ODIN BEGIN - APP", psize)
    data, labels = load_app()

    mx_label = np.max(labels)

    # data, labels = my_utils.load_known_applications()
    # data_u, labels_u = my_utils.load_unknown_applications()
    # _, data, _, labels = split(data, labels)
    # labels_u[labels_u > -1] = np.max(labels) + 1
    # data = np.concatenate([data, data_u])
    # labels = np.concatenate([labels, labels_u])

    model = my_utils.get_application_model("tensorflow")

    # need to search ideal eps

    og_argmax_predictions = np.argmax(model.predict(data), axis=-1)
    p_data = my_utils.create_perturbed_ds(data, model, psize)

    raw_predictions = model.predict(p_data)
    argmax_predictions = np.argmax(raw_predictions, axis=-1)
    max_predictions = np.max(raw_predictions, axis=-1)

    acc, recall, tn, tp, reject, scores, ood_r = [], [], [], [], [], [], []
    x = []
    best_acc = -1
    best_t = -1

    best_acc, best_t = plot_odin(max_predictions, argmax_predictions, labels, og_argmax_predictions,
                                 title="Application", save_name="odin_app")
    print("BEST ACCURACY ODIN APP:", best_acc)

    ood_idx = max_predictions < best_t
    new_predictions = np.copy(argmax_predictions)
    new_predictions[ood_idx] = np.max(labels)  # change labels to OOD when detected
    new_predictions[~ood_idx] = og_argmax_predictions[~ood_idx]

    d = my_utils.evaluate_without_model(new_predictions, labels, show_matrix=True, samples=True, use_torch=False,
                                        display_labels=display_labels_applications, excluded=True, title="odin_app_cf")
    print("TEST ODIN END")


def print_sizes():
    data, labels = my_utils.load_known_applications()
    data_u, labels_u = my_utils.load_unknown_applications()
    _, data, _, labels = split(data, labels)

    print("Applications: known -", labels.size, " unknown -", labels_u.size)

    data, labels = my_utils.load_known_malware()
    data_u, labels_u = my_utils.load_unknown_malware()
    _, data, _, labels = split(data, labels)
    _, data_u, _, labels_u = split(data_u, labels_u)

    print("Malware: known -", labels.size, " unknown -", labels_u.size)


def test_abstention_malware(data, labels):
    print("K+1/ABSTENTION TEST BEGIN - MALWARE")
    # data, labels = load_malware()
    # data, labels = my_utils.load_known_malware()
    # data_u, labels_u = my_utils.load_unknown_malware()
    # _, data, _, labels = split(data, labels)
    # _, data_u, _, labels_u = split(data_u, labels_u)
    #
    # labels_u[labels_u > -1] = np.max(labels) + 1
    #
    # data = np.concatenate([data, data_u])
    # labels = np.concatenate([labels, labels_u])

    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    model = my_utils.get_malware_model("tensorflow", k_plus_one=True)

    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, show_matrix=True,
                                    display_labels=display_labels_malware, use_torch=False, title="abstention_malware")
    # my_utils.new_evaluate(model, data, labels, show_matrix=True, display_labels=display_labels_malware)

    print("K+1/ABSTENTION TEST END - MALWARE")


def test_abstention_app(data, labels):
    print("K+1/ABSTENTION TEST BEGIN - APPLICATION")
    #
    # data, labels = my_utils.load_known_applications()
    # data_u, labels_u = my_utils.load_unknown_applications()
    # _, data, _, labels = split(data, labels)
    # _, data_u, _, labels_u = split(data_u, labels_u)
    #
    # labels_u[labels_u > -1] = np.max(labels) + 1
    #
    # data = np.concatenate([data, data_u])
    # labels = np.concatenate([labels, labels_u])
    #
    # data, labels = load_app()
    model = my_utils.get_application_model("tensorflow", k_plus_one=True)

    # my_utils.new_evaluate(model, data, labels, show_matrix=True, display_labels=display_labels_applications)
    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, show_matrix=True,
                                    display_labels=display_labels_applications, use_torch=False, title="abstention_app")
    print("K+1/ABSTENTION TEST END - APPLICATION")


def test_raw_with_ood():
    print("TEST BASE WITH OOD DATA- BEGIN")
    print("MALWARE")
    # data, labels = my_utils.load_known_malware()
    # data_u, labels_u = my_utils.load_unknown_malware()
    # _, data, _, labels = split(data, labels)
    # _, data_u, _, labels_u = split(data_u, labels_u)
    #
    # labels_u[labels_u > -1] = np.max(labels) + 1
    #
    # data = np.concatenate([data, data_u])
    # labels = np.concatenate([labels, labels_u])
    data, labels = load_malware()
    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    model = my_utils.get_malware_model("tensorflow")

    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, show_matrix=True,
                                    display_labels=display_labels_malware, use_torch=False, excluded=True, title="base_ood_malware_cf")

    # my_utils.new_evaluate(model, data, labels, show_matrix=True, display_labels=display_labels_malware)

    print("APPLICATION")
    data, labels = load_app()
    # data, labels = my_utils.load_known_applications()
    # data_u, labels_u = my_utils.load_unknown_applications(K1=False, training=False)
    # _, data, _, labels = split(data, labels)
    # labels_u[labels_u > -1] = np.max(labels) + 1
    # data = np.concatenate([data, data_u])
    # labels = np.concatenate([labels, labels_u])

    model = my_utils.get_application_model("tensorflow")
    # my_utils.new_evaluate(model, data, labels, show_matrix=True, display_labels=display_labels_applications)
    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, use_torch=False, show_matrix=True,
                                    display_labels=display_labels_applications, excluded=True, title="base_ood_app_cf")

    print("BASE WITH OOD DATA- END")


#


def test_raw():
    print("TEST RAW - BEGIN")
    print("MALWARE TF")
    data, labels = my_utils.load_known_malware()
    model = my_utils.get_malware_model("tensorflow")

    data = data.reshape(data.shape[0], data.shape[2], data.shape[1])
    _, data, _, labels = split(data, labels)

    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, show_matrix=True, use_torch=False,
                                    display_labels=display_labels_malware[:-1], ood=False, title="base_malware_cf")

    print("APPLICATION TF")
    data, labels = my_utils.load_known_applications()
    _, data, _, labels = split(data, labels)

    model = my_utils.get_application_model("tensorflow")

    my_utils.evaluate_without_model(np.argmax(model(data), axis=-1), labels, show_matrix=True, use_torch=False,
                                    display_labels=display_labels_applications[:-1], ood=False, title="base_app_cf")

    print("TEST RAW - END")





# def find_odin():
#     test_odin_app(0.01)
#     test_odin_app(0.04)
#     test_odin_app(0.1)
#
#     test_odin_app(0.4)
#     test_odin_app(0.9)

# find_odin()
app_data, app_labels = load_app()
mal_data, mal_labels = load_malware()
#
test_raw()
test_raw_with_ood()

test_abstention_app(app_data, app_labels)
test_odin_app(app_data, app_labels)
test_bp_app(app_data, app_labels)

test_abstention_malware(mal_data, mal_labels)
test_odin_malware(mal_data, mal_labels)
test_bp_malware(mal_data, mal_labels)
