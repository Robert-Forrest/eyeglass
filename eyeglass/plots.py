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


def plot_ternary(
        elements, model,
        originalData=None,
        quaternary=None,
        y_features=[]):

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

    compositions, percentages, all_features, GFA_predictions, realData, step = generate_ternary_compositions(
        elements, model, originalData, quaternary=quaternary, y_features=y_features)

    for feature in all_features.columns:

        if len(y_features) > 0:
            if feature not in y_features:
                continue

        save_path = ""
        if feature in cb.conf.target_names:
            save_path += ternary_dir + "/predictions/" + feature
            if not os.path.exists(ternary_dir + '/predictions'):
                os.makedirs(ternary_dir + '/predictions')
        else:
            save_path += ternary_dir + "/features/" + feature
            if not os.path.exists(ternary_dir + '/features'):
                os.makedirs(ternary_dir + '/features')

        title = None
        if quaternary is not None:
            title = "(" + "".join(elements) + ")$_{" + str(
                round(100 - quaternary[1], 2)) + "}$" + quaternary[0] + "$_{" + str(
                    round(quaternary[1], 2)) + "}$"

        if feature == 'GFA':

            mg.plots.ternary_heatmap(compositions, [p[0] for p in GFA_predictions],
                                     step, save_path=save_path+'_crystal',
                                     label="Crystal probability (%)",
                                     title=title, quaternary=quaternary, vmin=0, vmax=1)

            mg.plots.ternary_heatmap(compositions, [p[1] for p in GFA_predictions],
                                     step, save_path=save_path+'_GR',
                                     label="GR probability (%)",
                                     title=title, quaternary=quaternary, vmin=0, vmax=1)

            mg.plots.ternary_heatmap(compositions, [p[2] for p in GFA_predictions],
                                     step, save_path=save_path+'_BMG',
                                     label="BMG probability (%)",
                                     title=title, quaternary=quaternary, vmin=0, vmax=1)

            mg.plots.ternary_heatmap(compositions, [p[1]+p[2] for p in GFA_predictions],
                                     step, save_path=save_path+'_glass',
                                     label="Glass probability (%)",
                                     title=title, quaternary=quaternary, vmin=0, vmax=1)

            mg.plots.ternary_heatmap(compositions, [np.argmax(p) for p in GFA_predictions],
                                     step, save_path=save_path+'_argmax',
                                     label="Predicted class",
                                     title=title, quaternary=quaternary, vmin=0, vmax=2)

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
        elements, model, originalData, quaternary=None, minPercent=0, maxPercent=1, step=None, y_features=[]):
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
                    tmpComposition.append(100*(parsedComposition[e] / step))
                else:
                    tmpComposition.append(0)

            row['percentages'] = tuple(tmpComposition)
            realData.append(row)
    realData = pd.DataFrame(realData)

    compositions, percentages = mg.ternary.generate_alloys(
        elements, step*100, minPercent*100, maxPercent*100, quaternary)

    all_features = pd.DataFrame(compositions, columns=['composition'])
    all_features = cb.features.calculate_features(all_features,
                                                  dropCorrelatedFeatures=False,
                                                  plot=False,
                                                  additionalFeatures=y_features)
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

    if quaternary is not None:
        compositions, percentages = mg.ternary.generate_alloys(
            elements, step*100, minPercent*100, maxPercent*100)

    return compositions, percentages, all_features, GFA_predictions, realData, step


def plot_quaternary(elements, model, onlyPredictions=False, originalData=None, y_features=[]):
    if not os.path.exists(cb.conf.image_directory + "compositions"):
        os.makedirs(cb.conf.image_directory + "compositions")

    quaternary_dir = cb.conf.image_directory + "compositions/" + \
        "_".join(elements)

    if not os.path.exists(quaternary_dir):
        os.makedirs(quaternary_dir)

    minPercent = 0
    maxPercent = 100
    numPercentages = 6
    step = (maxPercent - minPercent) / float(numPercentages)
    quaternary_percentages = [round(percentage, 3)
                              for percentage in np.arange(minPercent, maxPercent, step)]

    heatmaps = {}
    compositions, percentages, all_features, GFA_predictions, realData, step = generate_ternary_compositions(
        elements[:3], model, originalData, quaternary=[elements[3], quaternary_percentages[0]], y_features=y_features)
    for feature in all_features.columns:
        trueFeatureName = feature.split(
            '_linearmix')[0].split('_discrepancy')[0]
        if onlyPredictions and feature not in cb.conf.target_names and trueFeatureName not in y_features:
            continue
        if feature not in heatmaps:
            if feature != 'GFA':
                heatmaps[feature] = []
            else:
                heatmaps['GFA_crystal'] = []
                heatmaps['GFA_glass'] = []
                heatmaps['GFA_GR'] = []
                heatmaps['GFA_BMG'] = []
                heatmaps['GFA_argmax'] = []

    realDatas = {}
    quaternary_compositions = []
    for percentage in quaternary_percentages:
        compositions, percentages, all_features, GFA_predictions, realData, step = generate_ternary_compositions(
            elements[:3], model, originalData, quaternary=[elements[3], percentage], y_features=y_features)

        realDatas[percentage] = realData
        quaternary_compositions.append(compositions)

        doneGFA = False
        for feature in heatmaps:
            if "GFA_" in feature:
                if not doneGFA:
                    doneGFA = True

                    heatmaps['GFA_crystal'].append(
                        [p[0] for p in GFA_predictions])
                    heatmaps['GFA_glass'].append(
                        [p[1]+p[2] for p in GFA_predictions])
                    heatmaps['GFA_GR'].append([p[1] for p in GFA_predictions])
                    heatmaps['GFA_BMG'].append([p[1] for p in GFA_predictions])
                    heatmaps['GFA_argmax'].append(
                        [np.argmax(p) for p in GFA_predictions])
            else:
                heatmaps[feature].append(all_features[feature])

    for feature in heatmaps:
        if 'GFA_' not in feature:
            vmax = -np.inf
            vmin = np.inf
            label = cb.features.prettyName(feature)
            if feature in cb.features.units:
                label += " (" + cb.features.units[feature] + ")"

            tmpFeature = feature
            suffix = None
            for i in range(len(quaternary_percentages)):
                for value in heatmaps[feature][i]:
                    if value > vmax:
                        vmax = value
                    if value < vmin:
                        vmin = value
        else:
            if feature != 'GFA_argmax':
                vmax = 1
                vmin = 0
                GFA_type = feature.split('_')[1]
                if GFA_type == 'BMG' or GFA_type == 'GR':
                    GFA_type = GFA_type.upper()
                else:
                    GFA_type = GFA_type[0].upper() + GFA_type[1:]

                label = GFA_type + " probability (%)"
                suffix = feature.split('_')[1]
                tmpFeature = "GFA"
            else:
                vmax = 2
                vmin = 0
                GFA_type = "GFA Classification"
                label = GFA_type
                suffix = feature.split('_')[1]
                tmpFeature = "GFA"

        for i in range(len(quaternary_percentages)):
            ternary_dir = cb.conf.image_directory + "compositions/" + \
                "_".join(elements) + "/" + \
                elements[3] + str(quaternary_percentages[i])
            if not os.path.exists(ternary_dir):
                os.makedirs(ternary_dir)

            save_path = ""
            if feature in cb.conf.target_names:
                save_path += ternary_dir + "/predictions/" + feature
                if not os.path.exists(ternary_dir + '/predictions'):
                    os.makedirs(ternary_dir + '/predictions')
            else:
                save_path += ternary_dir + "/features/" + feature
                if not os.path.exists(ternary_dir + '/features'):
                    os.makedirs(ternary_dir + '/features')

            title = "(" + "".join(elements[:-1]) + ")$_{" + str(
                round(100 - quaternary_percentages[i], 2)) + "}$" + elements[-1] + "$_{" + str(
                    round(quaternary_percentages[i], 2)) + "}$"

            scatter_data = []
            if(len(realDatas[quaternary_percentages[i]]) > 0):
                classes = ['Crystal', 'GR', 'BMG']
                markers = ['s', 'D', 'o']
                for c in range(len(classes)):
                    scatter_data.append({
                        'data': realDatas[quaternary_percentages[i]][realDatas[quaternary_percentages[i]]['GFA'] == c]['percentages'],
                        'label': classes[c],
                        'marker': markers[c]
                    })

            mg.plots.ternary_heatmap(quaternary_compositions[i], heatmaps[feature][i],
                                     step, save_path=save_path,
                                     scatter_data=scatter_data, title=title,
                                     label=label, vmax=vmax, vmin=vmin)

        columns = int(np.ceil(np.sqrt(len(quaternary_percentages))))
        rows = int(np.ceil(len(quaternary_percentages) / columns))
        numGridCells = columns*rows
        gridExcess = numGridCells - len(quaternary_percentages)

        fig = plt.figure(figsize=(4 * columns, 4 * rows))

        lastAx = None
        for i in reversed(range(len(quaternary_percentages))):
            iRow = i // columns
            iCol = i % columns

            ax = plt.subplot2grid(
                (rows, columns), (iRow, iCol))

            if gridExcess != 0 and iRow == (rows-1):
                if gridExcess % 2 == 0:
                    ax = plt.subplot2grid(
                        (rows, columns), (iRow, iCol+1))
                else:
                    ax = plt.subplot2grid(
                        (rows, columns * 2), (iRow, 1+(iCol*2)), colspan=2)

            if lastAx is None:
                lastAx = ax

            scatter_data = []
            if(len(realDatas[quaternary_percentages[i]]) > 0):
                classes = ['Crystal', 'GR', 'BMG']
                markers = ['s', 'D', 'o']
                for c in range(len(classes)):
                    scatter_data.append({
                        'data': realDatas[quaternary_percentages[i]][realDatas[quaternary_percentages[i]]['GFA'] == c]['percentages'],
                        'label': classes[c],
                        'marker': markers[c]
                    })

            title = "(" + "".join(elements[:-1]) + ")$_{" + str(
                round(100 - quaternary_percentages[i], 2)) + "}$" + elements[-1] + "$_{" + str(
                    round(quaternary_percentages[i], 2)) + "}$"

            mg.plots.ternary_heatmap(quaternary_compositions[i], heatmaps[feature][i],
                                     step, ax=ax, vmin=vmin, vmax=vmax,
                                     scatter_data=scatter_data, title=title,
                                     label=label, showColorbar=False)

        jet_cmap = mpl.cm.get_cmap('jet')

        cax = fig.add_axes([lastAx.get_position().x1 + 0.01,
                            lastAx.get_position().y0 + 0.03,
                            0.0075,
                            lastAx.get_position().height])

        colorbar = fig.colorbar(mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=jet_cmap),
            cax=cax)
        colorbar.set_label(label, labelpad=20, rotation=270)

        if feature in cb.conf.target_names or "GFA_" in feature:
            if not os.path.exists(quaternary_dir + '/predictions'):
                os.makedirs(quaternary_dir + '/predictions')

            fig.savefig(quaternary_dir + '/predictions/' +
                        feature + ".png", bbox_inches='tight')

        else:
            if not os.path.exists(quaternary_dir + '/features'):
                os.makedirs(quaternary_dir + '/features')

            fig.savefig(quaternary_dir + '/features/' +
                        feature + ".png", bbox_inches='tight')

        plt.clf()
        plt.cla()
        plt.close()
