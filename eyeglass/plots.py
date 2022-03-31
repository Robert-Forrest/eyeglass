import os

import ternary
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import metallurgy as mg
import cerebral as cb

mpl.use('Agg')
plt.style.use('ggplot')
plt.rc('axes', axisbelow=True)


def plot_binary(elements, model, originalData=None, x_features=["percentage"], y_Features=[]):

    if not os.path.exists(cb.conf.image_directory + "compositions"):
        os.makedirs(cb.conf.image_directory + "compositions")

    binary_dir = cb.conf.image_directory + \
        "compositions/" + "_".join(elements)
    if not os.path.exists(binary_dir):
        os.makedirs(binary_dir)

    realData = []
    requiredFeatures = []
    if originalData is not None:
        for _, row in originalData.iterrows():
            parsedComposition = mg.alloy.parse_composition(row['composition'])
            if set(elements).issuperset(set(parsedComposition.keys())):
                if elements[0] in parsedComposition:
                    row['percentage'] = parsedComposition[elements[0]] * 100
                else:
                    row['percentage'] = 0
                realData.append(row)
        realData = pd.DataFrame(realData)
        realData = realData.reset_index(drop=True)
        requiredFeatures = list(originalData.columns)

    for feature in x_features:
        if feature not in requiredFeatures:
            requiredFeatures.append(feature)

    compositions, percentages = mg.binary.generate_alloys(elements)

    all_features = pd.DataFrame(compositions, columns=['composition'])
    all_features = cb.features.calculate_features(all_features,
                                                  dropCorrelatedFeatures=False,
                                                  plot=False,
                                                  requiredFeatures=requiredFeatures,
                                                  additionalFeatures=y_features)
    all_features = all_features.drop('composition', axis='columns')
    all_features = all_features.fillna(cb.features.maskValue)
    for feature in cb.conf.targets:
        all_features[feature.name] = cb.features.maskValue
    all_features['GFA'] = all_features['GFA'].astype('int64')
    all_features['percentage'] = percentages
    GFA_predictions = []

    prediction_ds = cb.features.df_to_dataset(all_features)
    predictions = model.predict(prediction_ds)
    for i in range(len(cb.conf.targets)):
        if cb.conf.targets[i].name == 'GFA':
            GFA_predictions = predictions[i]
        else:
            all_features[cb.conf.targets[i].name] = predictions[i]

    for inspect_feature in x_features:
        if inspect_feature not in all_features.columns:
            continue
        if not os.path.exists(binary_dir+'/'+inspect_feature):
            os.makedirs(binary_dir + '/'+inspect_feature)

        for feature in all_features.columns:

            trueFeatureName = feature.split(
                '_linearmix')[0].split('_discrepancy')[0]

            if inspect_feature == feature:
                continue
            elif len(y_features) > 0:
                if feature not in y_features:
                    continue

            xlabel = None
            ylabel = None

            data = []
            labels = []
            scatter_data = None
            use_colorline = False

            if feature == 'GFA':
                crystal = []
                ribbon = []
                BMG = []
                for prediction in GFA_predictions:
                    crystal.append(prediction[0])
                    ribbon.append(prediction[1])
                    BMG.append(prediction[2])

                data.append(crystal)
                labels.append('Crystal')

                data.append(ribbon)
                labels.append('GR')

                data.append(BMG)
                labels.append('BMG')

                if len(realData) > 0 and inspect_feature in realData:
                    scatter_data = []
                    scatter_data.append({
                        'data': [
                            realData[realData['GFA'] == 0][inspect_feature],
                            [1] * len(realData[realData['GFA'] == 0]
                                      [inspect_feature])
                        ],
                        'marker': "s",
                        'label': None
                    })

                    scatter_data.append({
                        'data': [
                            realData[realData['GFA'] == 1][inspect_feature],
                            [1] * len(realData[realData['GFA'] == 1]
                                      [inspect_feature])
                        ],
                        'marker': "D",
                        'label': None
                    })

                    scatter_data.append({
                        'data': [
                            realData[realData['GFA'] == 2][inspect_feature],
                            [1] * len(realData[realData['GFA'] == 2]
                                      [inspect_feature])
                        ],
                        'marker': "o",
                        'label': None
                    })

            else:

                if len(realData) > 0 and feature in realData and inspect_feature in realData:
                    crystalData, crystalPercentages = cb.features.filter_masked(
                        realData[realData['GFA'] == 0][feature], realData[realData['GFA'] == 0][inspect_feature])
                    ribbonData, ribbonPercentages = cb.features.filter_masked(
                        realData[realData['GFA'] == 1][feature], realData[realData['GFA'] == 1][inspect_feature])
                    bmgData, bmgPercentages = cb.features.filter_masked(
                        realData[realData['GFA'] == 2][feature], realData[realData['GFA'] == 2][inspect_feature])

                    scatter_data = []
                    if len(crystalData) > 0:
                        if len(ribbonData) > 0 or len(bmgData) > 0:
                            scatter_data.append({
                                'data': [crystalPercentages, crystalData],
                                'marker': "s", 'label': "Crystal"
                            })
                    if len(ribbonData) > 0:
                        scatter_data.append({
                            'data': [ribbonPercentages, ribbonData],
                            'marker': "D", 'label': "GR"
                        })
                    if len(bmgData) > 0:
                        scatter_data.append({
                            'data': [bmgPercentages, bmgData],
                            'marker': "o", 'label': "BMG"
                        })

            if inspect_feature != 'percentage':
                xlabel = cb.features.prettyName(inspect_feature)
                if inspect_feature in cb.features.units:
                    xlabel += ' (' + cb.features.units[inspect_feature] + ')'

                use_colorline = True
                data = [all_features[inspect_feature], all_features[feature]]
            else:
                data = all_features[feature]

            if feature in cb.features.units:
                ylabel = cb.features.prettyName(
                    feature) + ' (' + cb.features.units[feature] + ')'
            elif feature == 'GFA':
                ylabel = "GFA Classification Confidence"
            else:
                ylabel = cb.features.prettyName(feature)

            save_path = ""
            if feature in cb.conf.target_names:
                save_path = binary_dir+'/'+inspect_feature + "/predictions/" + feature + ".png"
                if not os.path.exists(binary_dir+'/'+inspect_feature + '/predictions'):
                    os.makedirs(binary_dir+'/' +
                                inspect_feature + '/predictions')
            else:
                save_path = binary_dir+'/'+inspect_feature + "/features/" + feature + ".png"
                if not os.path.exists(binary_dir+'/'+inspect_feature + '/features'):
                    os.makedirs(binary_dir+'/'+inspect_feature + '/features')

            mg.plots.binary(
                compositions,
                data,
                scatter_data=scatter_data,
                xlabel=xlabel,
                ylabel=ylabel,
                use_colorline=use_colorline,
                save_path=save_path
            )


def plot_ternary(elements, model, onlyPredictions=False,
                 originalData=None, quaternary=None, additionalFeatures=[]):
    if not os.path.exists(cb.conf.image_directory + "compositions"):
        os.makedirs(cb.conf.image_directory + "compositions")

    if quaternary is None:
        ternary_dir = cb.conf.image_directory + \
            "compositions/" + "_".join(elements)
    else:
        ternary_dir = cb.conf.image_directory + "compositions/" + \
            "_".join(elements) + "_" + \
            quaternary[0] + "/" + quaternary[0] + str(quaternary[1])

    if not os.path.exists(ternary_dir):
        os.makedirs(ternary_dir)
    if not os.path.exists(ternary_dir + '/predictions'):
        os.makedirs(ternary_dir + '/predictions')
    if not os.path.exists(ternary_dir + '/features'):
        os.makedirs(ternary_dir + '/features')
        
    compositions, percentages, all_features, GFA_predictions, realData, step = generate_ternary_compositions(
        elements, model, originalData, quaternary=quaternary, additionalFeatures=additionalFeatures)

    for feature in all_features.columns:
        trueFeatureName = feature.split(
            '_linearmix')[0].split('_discrepancy')[0]
        if onlyPredictions and feature not in cb.conf.target_names and trueFeatureName not in additionalFeatures:
            continue

        save_path = ""
        if feature in cb.conf.target_names:
            save_path += ternary_dir + "/predictions/" + feature
        else:
            save_path += ternary_dir + "/features/" + feature

        title = None
        if quaternary is not None:
            title = "(" + "".join(elements) + ")$_{" + str(
                round(100 - quaternary[1], 2)) + "}$" + quaternary[0] + "$_{" + str(
                    round(quaternary[1], 2)) + "}$"

            
        if feature == 'GFA':
            heatmap_data_crystal, heatmap_data_glass, heatmap_data_ribbon, heatmap_data_bmg, heatmap_data_argmax = generate_ternary_heatmap_data(
                feature, GFA_predictions, percentages, step)

            mg.plots.ternary_heatmap(heatmap_data_crystal, feature, elements,
                            step, ternary_dir, realData=realData, quaternary=quaternary,
                                     label="Crystal probability (%)", title=title, suffix="Crystal", vmin=0, vmax=1)

            mg.plots.ternary_heatmap(heatmap_data_glass, feature, elements,
                            step, ternary_dir, realData=realData, quaternary=quaternary,
                            label="Glass probability (%)",title=title,  suffix="Glass", vmin=0, vmax=1)

            mg.plots.ternary_heatmap(heatmap_data_ribbon, feature, elements,
                            step, ternary_dir, realData=realData, quaternary=quaternary,
                            label="Ribbon probability (%)", title=title, suffix="Ribbon", vmin=0, vmax=1)

            mg.plots.ternary_heatmap(heatmap_data_bmg, feature, elements,
                            step, ternary_dir, realData=realData, quaternary=quaternary,
                            label="BMG probability (%)", title=title, suffix="BMG", vmin=0, vmax=1)

            mg.plots.ternary_heatmap(heatmap_data_argmax, feature, elements,
                            step, ternary_dir, realData=realData, quaternary=quaternary,
                            label="Predicted GFA Class", title=title, suffix="Classification", vmin=0, vmax=2)

        else:
            
            label = cb.features.prettyName(feature)
            if feature in cb.features.units:
                label += " ("+cb.features.units[feature]+")"
                
            mg.plots.ternary_heatmap(compositions, all_features[feature],
                                     step, save_path=save_path, label=label,
                                     title=title, quaternary=quaternary)




def ternary_scatter(data, tax):

    if len(data) > 0:
        plotted = False
        if len(data[data['GFA'] == 0]) > 0:
            if len(data[data['GFA'] == 1]) > 0 and len(data[data['GFA'] == 2]) > 0:
                tax.scatter(data[data['GFA'] == 0]['percentages'],
                            marker='s', label="Crystal", edgecolors='k',
                            zorder=2)
                plotted = True

        if len(data[data['GFA'] == 1]) > 0:
            tax.scatter(data[data['GFA'] == 1]['percentages'],
                        label="Ribbon", marker='D', zorder=2, edgecolors='k')
            plotted = True

        if len(data[data['GFA'] == 2]) > 0:
            tax.scatter(data[data['GFA'] == 2]['percentages'],
                        marker='o', label="BMG", zorder=2, edgecolors='k')
            plotted = True

        if plotted:
            tax.legend(loc="upper right", handletextpad=0.1, frameon=False)


def generate_ternary_compositions(
        elements, model, originalData, quaternary=None, minPercent=0, maxPercent=100, step=None, additionalFeatures=[]):
    if step is None:
        step = 0.02 * (maxPercent - minPercent)

    realData = []
    for _, row in originalData.iterrows():
        parsedComposition = mg.alloy.parse_composition(row['composition'])
        if quaternary is not None:
            if quaternary[1] > 0:
                if quaternary[0] not in parsedComposition:
                    continue
                elif parsedComposition[quaternary[0]] != quaternary[1]:
                    continue

        if set(elements).issuperset(set(parsedComposition.keys())):

            tmpComposition = []
            for e in elements:
                if e in parsedComposition:
                    tmpComposition.append(parsedComposition[e] * 100 / step)
                else:
                    tmpComposition.append(0)

            row['percentages'] = tuple(tmpComposition)
            realData.append(row)
    realData = pd.DataFrame(realData)

    compositions, percentages = mg.ternary.generate_alloys(
        elements, step, minPercent, maxPercent, quaternary)

    all_features = pd.DataFrame(compositions, columns=['composition'])
    all_features = cb.features.calculate_features(all_features,
                                               dropCorrelatedFeatures=False,
                                               plot=False,
                                               additionalFeatures=additionalFeatures)
    all_features = all_features.drop('composition', axis='columns')
    all_features = all_features.fillna(cb.features.maskValue)

    for feature in cb.conf.targets:
        all_features[feature.name] = cb.features.maskValue
    all_features['GFA'] = all_features['GFA'].astype('int64')
    GFA_predictions = []

    prediction_ds = cb.features.df_to_dataset(all_features)
    predictions = model.predict(prediction_ds)
    for i in range(len(cb.conf.targets)):
        if cb.conf.targets[i].name == 'GFA':
            GFA_predictions = predictions[i]
        else:
            all_features[cb.conf.targets[i].name] = predictions[i].flatten()

    return compositions, percentages, all_features, GFA_predictions, realData, step


def generate_ternary_heatmap_data(feature, data, percentages, step):

    if feature == 'GFA':

        heatmap_data_crystal = dict()
        heatmap_data_ribbon = dict()
        heatmap_data_bmg = dict()
        heatmap_data_glass = dict()
        heatmap_data_argmax = dict()
        for i in range(len(data)):
            heatmap_data_crystal[(percentages[i][0] / step,
                                  percentages[i][1] / step)] = data[i][0]
            heatmap_data_ribbon[(percentages[i][0] / step,
                                 percentages[i][1] / step)] = data[i][1]
            heatmap_data_bmg[(percentages[i][0] / step,
                              percentages[i][1] / step)] = data[i][2]
            heatmap_data_glass[(percentages[i][0] / step,
                                percentages[i][1] / step)] = 1 - data[i][0]
            heatmap_data_argmax[(percentages[i][0] / step,
                                 percentages[i][1] / step)] = np.argmax(data[i])

        return heatmap_data_crystal, heatmap_data_glass, heatmap_data_ribbon, heatmap_data_bmg, heatmap_data_argmax
    else:
        heatmap_data = dict()
        for i, row in data.iterrows():
            heatmap_data[(percentages[i][0] / step,
                          percentages[i][1] / step)] = row[feature]
        return heatmap_data
