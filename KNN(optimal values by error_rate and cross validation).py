{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNnqQX5bSsI2QniFXb+88PM",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/TaimoorAbdullah/KNN/blob/main/Project_13.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "kVOq6Earyvaj"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import confusion_matrix\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.metrics import precision_score\n",
        "from sklearn.datasets import make_blobs\n",
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn.model_selection import cross_val_predict\n",
        "from sklearn.model_selection import KFold\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "from matplotlib import style\n",
        "style.use('dark_background')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x,y  =  make_blobs(n_samples = 1000,n_features = 3,centers = 3,cluster_std = 4,random_state = 42)\n",
        "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state  = 42)\n",
        "sc = StandardScaler()\n",
        "x_train =  sc.fit_transform(x_train)\n",
        "x_test = sc.transform(x_test)\n",
        "knn  =  KNeighborsClassifier()\n",
        "knn.fit(x_train,y_train)\n"
      ],
      "metadata": {
        "id": "zOJ8qHHm8b1U",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 80
        },
        "outputId": "3946a18b-c600-40af-b87f-02058baa6aa4"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "KNeighborsClassifier()"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {\n",
              "  /* Definition of color scheme common for light and dark mode */\n",
              "  --sklearn-color-text: #000;\n",
              "  --sklearn-color-text-muted: #666;\n",
              "  --sklearn-color-line: gray;\n",
              "  /* Definition of color scheme for unfitted estimators */\n",
              "  --sklearn-color-unfitted-level-0: #fff5e6;\n",
              "  --sklearn-color-unfitted-level-1: #f6e4d2;\n",
              "  --sklearn-color-unfitted-level-2: #ffe0b3;\n",
              "  --sklearn-color-unfitted-level-3: chocolate;\n",
              "  /* Definition of color scheme for fitted estimators */\n",
              "  --sklearn-color-fitted-level-0: #f0f8ff;\n",
              "  --sklearn-color-fitted-level-1: #d4ebff;\n",
              "  --sklearn-color-fitted-level-2: #b3dbfd;\n",
              "  --sklearn-color-fitted-level-3: cornflowerblue;\n",
              "\n",
              "  /* Specific color for light theme */\n",
              "  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
              "  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));\n",
              "  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));\n",
              "  --sklearn-color-icon: #696969;\n",
              "\n",
              "  @media (prefers-color-scheme: dark) {\n",
              "    /* Redefinition of color scheme for dark theme */\n",
              "    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
              "    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));\n",
              "    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));\n",
              "    --sklearn-color-icon: #878787;\n",
              "  }\n",
              "}\n",
              "\n",
              "#sk-container-id-1 {\n",
              "  color: var(--sklearn-color-text);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 pre {\n",
              "  padding: 0;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 input.sk-hidden--visually {\n",
              "  border: 0;\n",
              "  clip: rect(1px 1px 1px 1px);\n",
              "  clip: rect(1px, 1px, 1px, 1px);\n",
              "  height: 1px;\n",
              "  margin: -1px;\n",
              "  overflow: hidden;\n",
              "  padding: 0;\n",
              "  position: absolute;\n",
              "  width: 1px;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-dashed-wrapped {\n",
              "  border: 1px dashed var(--sklearn-color-line);\n",
              "  margin: 0 0.4em 0.5em 0.4em;\n",
              "  box-sizing: border-box;\n",
              "  padding-bottom: 0.4em;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-container {\n",
              "  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`\n",
              "     but bootstrap.min.css set `[hidden] { display: none !important; }`\n",
              "     so we also need the `!important` here to be able to override the\n",
              "     default hidden behavior on the sphinx rendered scikit-learn.org.\n",
              "     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */\n",
              "  display: inline-block !important;\n",
              "  position: relative;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-text-repr-fallback {\n",
              "  display: none;\n",
              "}\n",
              "\n",
              "div.sk-parallel-item,\n",
              "div.sk-serial,\n",
              "div.sk-item {\n",
              "  /* draw centered vertical line to link estimators */\n",
              "  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));\n",
              "  background-size: 2px 100%;\n",
              "  background-repeat: no-repeat;\n",
              "  background-position: center center;\n",
              "}\n",
              "\n",
              "/* Parallel-specific style estimator block */\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item::after {\n",
              "  content: \"\";\n",
              "  width: 100%;\n",
              "  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);\n",
              "  flex-grow: 1;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel {\n",
              "  display: flex;\n",
              "  align-items: stretch;\n",
              "  justify-content: center;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  position: relative;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item {\n",
              "  display: flex;\n",
              "  flex-direction: column;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item:first-child::after {\n",
              "  align-self: flex-end;\n",
              "  width: 50%;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item:last-child::after {\n",
              "  align-self: flex-start;\n",
              "  width: 50%;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-parallel-item:only-child::after {\n",
              "  width: 0;\n",
              "}\n",
              "\n",
              "/* Serial-specific style estimator block */\n",
              "\n",
              "#sk-container-id-1 div.sk-serial {\n",
              "  display: flex;\n",
              "  flex-direction: column;\n",
              "  align-items: center;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  padding-right: 1em;\n",
              "  padding-left: 1em;\n",
              "}\n",
              "\n",
              "\n",
              "/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is\n",
              "clickable and can be expanded/collapsed.\n",
              "- Pipeline and ColumnTransformer use this feature and define the default style\n",
              "- Estimators will overwrite some part of the style using the `sk-estimator` class\n",
              "*/\n",
              "\n",
              "/* Pipeline and ColumnTransformer style (default) */\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable {\n",
              "  /* Default theme specific background. It is overwritten whether we have a\n",
              "  specific estimator or a Pipeline/ColumnTransformer */\n",
              "  background-color: var(--sklearn-color-background);\n",
              "}\n",
              "\n",
              "/* Toggleable label */\n",
              "#sk-container-id-1 label.sk-toggleable__label {\n",
              "  cursor: pointer;\n",
              "  display: flex;\n",
              "  width: 100%;\n",
              "  margin-bottom: 0;\n",
              "  padding: 0.5em;\n",
              "  box-sizing: border-box;\n",
              "  text-align: center;\n",
              "  align-items: start;\n",
              "  justify-content: space-between;\n",
              "  gap: 0.5em;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 label.sk-toggleable__label .caption {\n",
              "  font-size: 0.6rem;\n",
              "  font-weight: lighter;\n",
              "  color: var(--sklearn-color-text-muted);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 label.sk-toggleable__label-arrow:before {\n",
              "  /* Arrow on the left of the label */\n",
              "  content: \"▸\";\n",
              "  float: left;\n",
              "  margin-right: 0.25em;\n",
              "  color: var(--sklearn-color-icon);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {\n",
              "  color: var(--sklearn-color-text);\n",
              "}\n",
              "\n",
              "/* Toggleable content - dropdown */\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content {\n",
              "  max-height: 0;\n",
              "  max-width: 0;\n",
              "  overflow: hidden;\n",
              "  text-align: left;\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content.fitted {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content pre {\n",
              "  margin: 0.2em;\n",
              "  border-radius: 0.25em;\n",
              "  color: var(--sklearn-color-text);\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-toggleable__content.fitted pre {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {\n",
              "  /* Expand drop-down */\n",
              "  max-height: 200px;\n",
              "  max-width: 100%;\n",
              "  overflow: auto;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {\n",
              "  content: \"▾\";\n",
              "}\n",
              "\n",
              "/* Pipeline/ColumnTransformer-specific style */\n",
              "\n",
              "#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Estimator-specific style */\n",
              "\n",
              "/* Colorize estimator box */\n",
              "#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-label label.sk-toggleable__label,\n",
              "#sk-container-id-1 div.sk-label label {\n",
              "  /* The background is the default theme color */\n",
              "  color: var(--sklearn-color-text-on-default-background);\n",
              "}\n",
              "\n",
              "/* On hover, darken the color of the background */\n",
              "#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "/* Label box, darken color on hover, fitted */\n",
              "#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {\n",
              "  color: var(--sklearn-color-text);\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Estimator label */\n",
              "\n",
              "#sk-container-id-1 div.sk-label label {\n",
              "  font-family: monospace;\n",
              "  font-weight: bold;\n",
              "  display: inline-block;\n",
              "  line-height: 1.2em;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-label-container {\n",
              "  text-align: center;\n",
              "}\n",
              "\n",
              "/* Estimator-specific */\n",
              "#sk-container-id-1 div.sk-estimator {\n",
              "  font-family: monospace;\n",
              "  border: 1px dotted var(--sklearn-color-border-box);\n",
              "  border-radius: 0.25em;\n",
              "  box-sizing: border-box;\n",
              "  margin-bottom: 0.5em;\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-0);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-estimator.fitted {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-0);\n",
              "}\n",
              "\n",
              "/* on hover */\n",
              "#sk-container-id-1 div.sk-estimator:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-2);\n",
              "}\n",
              "\n",
              "#sk-container-id-1 div.sk-estimator.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-2);\n",
              "}\n",
              "\n",
              "/* Specification for estimator info (e.g. \"i\" and \"?\") */\n",
              "\n",
              "/* Common style for \"i\" and \"?\" */\n",
              "\n",
              ".sk-estimator-doc-link,\n",
              "a:link.sk-estimator-doc-link,\n",
              "a:visited.sk-estimator-doc-link {\n",
              "  float: right;\n",
              "  font-size: smaller;\n",
              "  line-height: 1em;\n",
              "  font-family: monospace;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  border-radius: 1em;\n",
              "  height: 1em;\n",
              "  width: 1em;\n",
              "  text-decoration: none !important;\n",
              "  margin-left: 0.5em;\n",
              "  text-align: center;\n",
              "  /* unfitted */\n",
              "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-unfitted-level-1);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link.fitted,\n",
              "a:link.sk-estimator-doc-link.fitted,\n",
              "a:visited.sk-estimator-doc-link.fitted {\n",
              "  /* fitted */\n",
              "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-fitted-level-1);\n",
              "}\n",
              "\n",
              "/* On hover */\n",
              "div.sk-estimator:hover .sk-estimator-doc-link:hover,\n",
              ".sk-estimator-doc-link:hover,\n",
              "div.sk-label-container:hover .sk-estimator-doc-link:hover,\n",
              ".sk-estimator-doc-link:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,\n",
              ".sk-estimator-doc-link.fitted:hover,\n",
              "div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,\n",
              ".sk-estimator-doc-link.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "/* Span, style for the box shown on hovering the info icon */\n",
              ".sk-estimator-doc-link span {\n",
              "  display: none;\n",
              "  z-index: 9999;\n",
              "  position: relative;\n",
              "  font-weight: normal;\n",
              "  right: .2ex;\n",
              "  padding: .5ex;\n",
              "  margin: .5ex;\n",
              "  width: min-content;\n",
              "  min-width: 20ex;\n",
              "  max-width: 50ex;\n",
              "  color: var(--sklearn-color-text);\n",
              "  box-shadow: 2pt 2pt 4pt #999;\n",
              "  /* unfitted */\n",
              "  background: var(--sklearn-color-unfitted-level-0);\n",
              "  border: .5pt solid var(--sklearn-color-unfitted-level-3);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link.fitted span {\n",
              "  /* fitted */\n",
              "  background: var(--sklearn-color-fitted-level-0);\n",
              "  border: var(--sklearn-color-fitted-level-3);\n",
              "}\n",
              "\n",
              ".sk-estimator-doc-link:hover span {\n",
              "  display: block;\n",
              "}\n",
              "\n",
              "/* \"?\"-specific style due to the `<a>` HTML tag */\n",
              "\n",
              "#sk-container-id-1 a.estimator_doc_link {\n",
              "  float: right;\n",
              "  font-size: 1rem;\n",
              "  line-height: 1em;\n",
              "  font-family: monospace;\n",
              "  background-color: var(--sklearn-color-background);\n",
              "  border-radius: 1rem;\n",
              "  height: 1rem;\n",
              "  width: 1rem;\n",
              "  text-decoration: none;\n",
              "  /* unfitted */\n",
              "  color: var(--sklearn-color-unfitted-level-1);\n",
              "  border: var(--sklearn-color-unfitted-level-1) 1pt solid;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 a.estimator_doc_link.fitted {\n",
              "  /* fitted */\n",
              "  border: var(--sklearn-color-fitted-level-1) 1pt solid;\n",
              "  color: var(--sklearn-color-fitted-level-1);\n",
              "}\n",
              "\n",
              "/* On hover */\n",
              "#sk-container-id-1 a.estimator_doc_link:hover {\n",
              "  /* unfitted */\n",
              "  background-color: var(--sklearn-color-unfitted-level-3);\n",
              "  color: var(--sklearn-color-background);\n",
              "  text-decoration: none;\n",
              "}\n",
              "\n",
              "#sk-container-id-1 a.estimator_doc_link.fitted:hover {\n",
              "  /* fitted */\n",
              "  background-color: var(--sklearn-color-fitted-level-3);\n",
              "}\n",
              "</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>KNeighborsClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator fitted sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label fitted sk-toggleable__label-arrow\"><div><div>KNeighborsClassifier</div></div><div><a class=\"sk-estimator-doc-link fitted\" rel=\"noreferrer\" target=\"_blank\" href=\"https://scikit-learn.org/1.6/modules/generated/sklearn.neighbors.KNeighborsClassifier.html\">?<span>Documentation for KNeighborsClassifier</span></a><span class=\"sk-estimator-doc-link fitted\">i<span>Fitted</span></span></div></label><div class=\"sk-toggleable__content fitted\"><pre>KNeighborsClassifier()</pre></div> </div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "np.unique(y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oS3vAnTLH59Y",
        "outputId": "ca979e61-755b-47b4-bebb-354c51c325ae"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 1, 2])"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize = (7,7))\n",
        "plt.scatter(x_train[y_train  ==0,0],x_train[y_train ==0,1],s = 15)\n",
        "plt.scatter(x_train[y_train ==1,0],x_train[y_train==1,1],s = 15)\n",
        "plt.scatter(x_train[y_train==2,0],x_train[y_train==2,1],s = 15)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 599
        },
        "id": "uQ3KYLS8GHmx",
        "outputId": "3bac2864-6810-4363-9087-625b4c689742"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 700x700 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize  =  (5,5))\n",
        "plt.scatter(x_test[y_test == 0,0],x_test[y_test==0,1],s = 15,color = 'yellow')\n",
        "plt.scatter(x_test[y_test==1,0],x_test[y_test==1,1],s=15,color = 'red')\n",
        "plt.scatter(x_test[y_test ==2,0],x_test[y_test==2,1],s = 15,color = 'blue')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 445
        },
        "id": "M0TVWM2-I6Gt",
        "outputId": "3637171e-e7a4-4137-8f1e-0f06fb761e49"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 500x500 with 1 Axes>"
            ],
            "image/png": "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/co+Y4xef/11rV271mLLwAeuvvpqSeJeMou+vj59+9vf1uDgoA4dOpT475yv0u8yN954o3784x9r+/bttpsCllm6dKn6+/t18uTJOftPnjypVatWWWoV+EClUtGvfvUrHTx4UH/+859tN8c6t956qw4dOqRPfepT+te//qX77rtPf/nLXxL/PZGapF27dskY01E333zznL8ZGhrS7373O7388svat2+fpZbnSzfnBQDS8fzzz+vWW2/Vd77zHdtNcYK//vWv+upXv6qvf/3r2rt3r1566SV9+ctfTvz3RGqSnnvuOb344osd33P8+PEr/33dddfpjTfe0NjYmH70ox/l3Dp7pD0vZeb06dOq1+tatmzZnP3Lli3TBx98YKlV4Dp79uzRt771LX3jG9/Q5OSk7eY4wfT0tN5++21J0p/+9Cfdcccd+slPfqKHHnoo0d9jaopuSKdPn0703qGhIb3xxhs6fPiwHnzwQRljcm6dPdKcl7IzPT2tw4cPa/369XrllVckRd1K69ev169//WvLrQMX2bNnj+677z6tW7dO7777ru3mOEtfX5+WLFmS+P2YWgqGhoY0Ojqq9957T9u3b5+Tqt06llI2Vq5cqc9+9rO6/vrrtWjRIq1Zs0aSdOzYMX388ceWW1cMu3fv1ksvvaQ//vGPeuutt/TII49ocHBQ+/fvt900awwODuqLX/zilddf+MIXtGbNGn344YeamJiw2DK7PP/883rggQe0efNmnTt37kqE/9FHH+nixYuWW2ePZ555Rr/97W/197//XdVqVQ888IDWrVunb37zm6k+x3rKpi/asmWLicN222xr//79bc/L8PCw9bYVqW3btpl3333XXLx40YyPj5s777zTeptsanh4uO3vYv/+/dbbZlNxbNmyxXrbbGrfvn3mnXfeMRcvXjQnT540Bw4cMBs2bEj1GaynBgAAwUD2IwAABAOmBgAAwYCpAQBAMGBqAAAQDJgaAAAEA6YGAADBgKkBAEAwYGoAABAMmBoAAAQDpgYAAMGAqQEAQDD8fxmK4i81J6FHAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "prediction  = knn.predict(x_test)\n",
        "accuracy = np.mean(prediction  ==y_test)\n",
        "accuracy\n"
      ],
      "metadata": {
        "id": "hba7marK8b6g",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ba25cc84-2c90-485d-ccdf-8aeca52c8de5"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "np.float64(0.835)"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(x_test[prediction==0,0],x_test[prediction==0,1],s = 15,color = 'yellow')\n",
        "plt.scatter(x_test[prediction==1,0],x_test[prediction==1,1],s = 15,color = 'red')\n",
        "plt.scatter(x_test[prediction==2,0] , x_test[prediction==2,1],s = 15,color = 'blue')\n"
      ],
      "metadata": {
        "id": "K0UIRdK08b8f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 447
        },
        "outputId": "ecdf8fa0-688c-47e4-ff53-8f634ee3c68e"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.collections.PathCollection at 0x7c6edeb99700>"
            ]
          },
          "metadata": {},
          "execution_count": 7
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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AAA4QyECAACcoRABAADOUIgAAABnKEQAAIAzFCIAAMAZChEAAOAMhQgAAHAmiGvNVCoV100AAAA9StJve12ItF7I7Oys45YAAICkKpWKFhYWun7PgCSTT3PSGR0d3fRF+KBSqWh2dlY7d+4Mor0h4hjng+NsH8c4Hxxn+7od40qlor///e+bPobXIyKSenoRPllYWOAP3jKOcT44zvZxjPPBcbav0zHu9ZgzWRUAADhDIQIAAJyhEMnIv//9bz3yyCP697//7bophcUxzgfH2T6OcT44zvZlcYy9n6wKAACKixERAADgDIUIAABwhkIEAAA4QyECAACcoRDJ2Jvf/GYdOXJEZ8+e1eXLl3XmzBk98sgj2rp1q+umFc7XvvY1vfDCC3rllVd08eJF180phM997nN66aWXdOXKFZ08eVK333676yYVzp133qmf/exnmp2dlTFG+/fvd92kwvnqV7+q3/72t7p06ZLm5ub0k5/8RO94xztcN6tQHnjgAZ0+fVovv/yyXn75ZdVqNd19992pHotCJGO33HKLBgcHdeDAAb373e/WF7/4RT3wwAP61re+5bpphTM8PKxnn31WTz75pOumFMJ9992nxx57TN/4xjf03ve+V6dPn9YvfvELXX/99a6bVijbtm3T6dOn9eCDD7puSmGNj4/riSee0B133KF9+/Zp69at+uUvf6nXvva1rptWGOfOndNXv/pV7d69W+973/v0q1/9Sj/96U/1rne9K9XjGWI3Bw8eNC+++KLzdhQ1999/v7l48aLzdoSekydPmscff/zq1wMDA+bcuXPmK1/5ivO2FTXGGLN//37n7Sh6duzYYYwx5s4773TeliLnn//8p/n0pz+d+OcYEcnBtddeqwsXLrhuBhBr69at2r17t5577rmr9xlj9Nxzz2nPnj0OWwb079prr5Uk3octGRwc1Ec/+lFt27ZNJ06cSPzz3l/0LnRvfetb9dBDD+ngwYOumwLE2rFjh4aGhjQ3N7fu/rm5Od1yyy2OWgX0b2BgQN///vf1m9/8Rn/84x9dN6dQbr31Vp04cUKvec1r9K9//Uv33nuv/vSnPyV+HEZEevToo4/KGNM1N99887qfGR0d1bFjx/Tss8/qyJEjjloeljTHGQDiPPHEE7r11lv1sY99zHVTCufPf/6z3vOe9+gDH/iAnnzyST3zzDN65zvfmfhxGBHp0fe+9z09/fTTXb/n7NmzV//7xhtv1PPPP69arabPfvazlltXHEmPM7IxPz+vRqOhG264Yd39N9xwg/7xj384ahXQn8cff1wf+chH9MEPflCzs7Oum1M4S0tLevHFFyVJv//973X77bfr85//vB544IFEj0Mh0qP5+XnNz8/39L2jo6N6/vnnderUKX3qU5+SMcZy64ojyXFGdpaWlnTq1Cnddddd+ulPfyopGtK+66679IMf/MBx64DkHn/8cd17772amJjQX/7yF9fNKYXBwUFdc801iX+OQiRjo6Ojmp6e1l//+lcdPHhw3dLH9vPv6M+uXbt03XXX6U1vepO2bNmi2267TZJ05swZvfLKK45bF57HHntMzzzzjH73u9/pt7/9rb7whS9o27ZtOnr0qOumFcq2bdv0tre97erXb3nLW3TbbbfpwoULqtfrDltWHE888YQ+/vGPa//+/VpYWLg60vfyyy/r1Vdfddy6YvjWt76ln//85/rb3/6mSqWij3/845qYmNCHP/zhVI/nfMlPkXL//febOK7bVrQcPXq043EeHx933rZQ8+CDD5q//OUv5tVXXzUnT54073//+523qWgZHx/v+Hd79OhR520rSuLcf//9zttWlBw5csS89NJL5tVXXzVzc3Pm+PHjZu/evakea6D5HwAAALlj1QwAAHCGQgQAADhDIQIAAJyhEAEAAM5QiAAAAGcoRAAAgDMUIgAAwBkKEQAA4AyFCAAAcIZCBAAAOEMhAgAAnKEQAQAAzvx/zUwfvmKuNfAAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "error_rate  =[]\n",
        "for i in range(1,30):\n",
        "  knn1   = KNeighborsClassifier(n_neighbors = i)\n",
        "  knn1.fit(x_train,y_train)\n",
        "  pred  = knn1.predict(x_test)\n",
        "  current_accuracy =  np.mean(pred ==y_test)\n",
        "  error_rate_value =  1-current_accuracy\n",
        "  error_rate.append(error_rate_value)"
      ],
      "metadata": {
        "id": "Lh_OSA4w8cAE"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "XGzSX247Os10"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plt.plot(range(1,30),error_rate,marker = 'o',markerfacecolor = 'blue',markeredgecolor = 'red',lw   = 1.5)"
      ],
      "metadata": {
        "id": "BAh9zui48cDL",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 447
        },
        "outputId": "bcad5b15-a013-4ba3-e982-152829a4c395"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7c6ede9a4470>]"
            ]
          },
          "metadata": {},
          "execution_count": 10
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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GMONN9541jbz589n//79nDp1ivfff59BgwZZkGlwOt/7t2zZsrOuyXfeeceibIPPrFmz+OyzzygtLaW4uJg33niDCy+8sME2kZGR/OY3v6GkpISysjJWrFhB7969Lco4+DTnPVyzZs1Z1+Hzzz9vUcbB5d577yU3N5fjx49z/PhxPv30UyZNmuRdH6jrr10XLtOmTWPx4sXMnz+f9PR0cnNzWb16NbGxsVanFhK2bt1KfHy8N6688kqrUwpq0dHR5Obmct999zW6/uGHH+a//uu/uPfeexk5ciQnT55k9erVREZGtnGmwel87x/AO++80+CavO2229oww+A2duxYli5dyqhRo5gwYQLh4eG89957REVFebd57rnnuP766/n+97/P2LFjSUxM5G9/+5uFWQeX5ryHAH/4wx8aXIcPP/ywRRkHl3379jFr1iwyMjK4/PLL+fDDD1m5ciUpKSlAYK8/014jOzvbLFmyxPu9zWYz+/btMzNnzrQ8t2CPefPmmU2bNlmeR6iGMcbceOONDZbt37/fPPDAA97vY2JiTHl5ubnlllsszzfYorH3b9myZeaNN96wPLdQiV69ehljjBkzZoyBuuutsrLSfO973/Nuc9FFFxljjBk5cqTl+QZjnPkeAmbNmjXmueeeszy3UInDhw+bu+++O6DXX7ttcQkPDycjI4OsrCzvMmMMWVlZjB492sLMQsfgwYMpLCxk165dvPTSS1xwwQVWpxSykpOTSUhIaHA9lpaWsn79el2PPrj66qspLi7myy+/5Le//S09evSwOqWg1bVrVwCOHDkCQEZGBhEREQ2uwR07drBnzx5dg0048z30uP322zl06BB5eXksWLCAzp07W5FeULPb7dxyyy1ER0fjcrkCev21q4csfluvXr1wOBwUFxc3WF5cXMyQIUMsyip0rF+/nh/+8Ifs2LGDhIQE5s2bx7p160hNTeXEiRNWpxdy4uPjARq9Hj3r5Nzeffdd/va3v1FQUMDAgQNZsGAB77zzDqNHj6a2ttbq9IKKzWbjV7/6FR9//DHbtm0D6q7ByspKjh8/3mBbXYONa+w9BHjllVfYs2cP+/fv59JLL+Xpp5/moosu4nvf+56F2QaP1NRUXC4XnTp14sSJE0ydOpXt27czbNiwgF1/7bZwEf+8++673n/n5eWxfv169uzZw7Rp0/jTn/5kYWbSUb322mvef2/dupUtW7aQn5/P1VdfzYcffmhhZsFn6dKlpKamql+aH5p6D1944QXvv7du3UpRUREffvghAwYMID8/v63TDDo7duxg2LBhdO3alZtvvpnly5czduzYgJ6j3d4qKikpwe12ExcX12B5XFwcBw4csCir0HX8+HG++uorjYJpIc81p+sxcAoKCjh06JCuyTMsWbKEKVOmMG7cOAoLC73LDxw4QGRkpPf2h4euwbM19R42Zv369QC6DutVV1eza9cucnJymDNnDrm5ucyYMSOg11+7LVyqq6vZuHEjmZmZ3mU2m43MzExcLpeFmYWm6OhoBg4cSFFRkdWphKSCggKKiooaXI9Op5ORI0fqemyhpKQkevbsqWvyW5YsWcLUqVO55ppr2L17d4N1GzdupKqqqsE1eOGFF9KvXz9dg99yrvewMcOGDQPQddgEu91OZGRkwK8/y3sdt1ZMmzbNlJeXmzvvvNMMGTLE/O53vzNHjhwxvXv3tjy3YI9nn33WXHXVVaZfv35m9OjR5r333jMHDx40vXr1sjy3YI3o6GiTlpZm0tLSjDHG3H///SYtLc1ccMEFBjAPP/ywOXLkiLn++utNamqqeeONN8yuXbtMZGSk5bkHQ5zr/YuOjjbPPPOMGTlypOnXr5+55pprzIYNG8yOHTtMRESE5bkHQyxdutQcPXrUXHXVVSYuLs4bnTp18m7z29/+1uzevdtcffXVJj093XzyySfmk08+sTz3YInzvYcDBgwwv/jFL0x6errp16+fuf76683OnTvN2rVrLc89GGLBggVmzJgxpl+/fiY1NdUsWLDA1NTUmPHjxxsI6PVn/YttzbjvvvvM7t27TUVFhcnOzjYjRoywPKdQiFdffdUUFhaaiooKs3fvXvPqq6+aAQMGWJ5XMMfYsWNNY5YtW+bdZv78+aaoqMiUl5eb999/3wwePNjyvIMlzvX+derUybz77rumuLjYVFZWmoKCAvP73/9ef4R8K5py1113ebeJjIw0v/nNb8zhw4fNiRMnzP/93/+ZuLg4y3MPljjfe9inTx+zdu1aU1JSYsrLy81XX31lnn76aeN0Oi3PPRjij3/8oykoKDAVFRWmuLjYvP/++96iBQJ3/dnq/yEiIiIS9NptHxcRERFpf1S4iIiISMhQ4SIiIiIhQ4WLiIiIhAwVLiIiIhIyVLiIiIhIyFDhIiIiIiFDhYuIiIiEDBUuIiIiEjJUuIiIiEjIUOEiIiIiIUOFi4iIiISM/w9NHAAMaE5jFQAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "knn2  =  KNeighborsClassifier(n_neighbors =  20)\\\n",
        "knn2.fit(x_train,y_train)\n",
        "prediction2  =  knn2.predict(x_test)\n",
        "accuracy3 = np.mean(prediction2 ==y_test)\n",
        "accuracy3"
      ],
      "metadata": {
        "id": "o5siYzpK8cF0",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "bfe3e958-1a4d-47ae-f789-3dd19c276a5a"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "np.float64(0.895)"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "cv =  KFold(n_splits  =  5,shuffle = True , random_state  =  42)\n",
        "model  =  KNeighborsClassifier()\n",
        "parameters = {'n_neighbors':[np.arange(1,30)]}\n",
        "grid  =  GridSearchCV(model, parameters,cv = cv)\n",
        "grid.fit(x_train,y_train)\n",
        "grid.best_params_"
      ],
      "metadata": {
        "id": "Yo4m0oCs8cIl",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 668
        },
        "outputId": "b1e6f1d4-0598-48bd-d5e8-a7cf8171a972"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ValueError",
          "evalue": "\nAll the 5 fits failed.\nIt is very likely that your model is misconfigured.\nYou can try to debug the error by setting error_score='raise'.\n\nBelow are more details about the failures:\n--------------------------------------------------------------------------------\n5 fits failed with the following error:\nTraceback (most recent call last):\n  File \"/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_validation.py\", line 866, in _fit_and_score\n    estimator.fit(X_train, y_train, **fit_params)\n  File \"/usr/local/lib/python3.12/dist-packages/sklearn/base.py\", line 1382, in wrapper\n    estimator._validate_params()\n  File \"/usr/local/lib/python3.12/dist-packages/sklearn/base.py\", line 436, in _validate_params\n    validate_parameter_constraints(\n  File \"/usr/local/lib/python3.12/dist-packages/sklearn/utils/_param_validation.py\", line 98, in validate_parameter_constraints\n    raise InvalidParameterError(\nsklearn.utils._param_validation.InvalidParameterError: The 'n_neighbors' parameter of KNeighborsClassifier must be an int in the range [1, inf) or None. Got array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,\n       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]) instead.\n",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipykernel_254/3473006000.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mparameters\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m{\u001b[0m\u001b[0;34m'n_neighbors'\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m30\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mgrid\u001b[0m  \u001b[0;34m=\u001b[0m  \u001b[0mGridSearchCV\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mparameters\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mcv\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcv\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 5\u001b[0;31m \u001b[0mgrid\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0my_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      6\u001b[0m \u001b[0mgrid\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbest_params_\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/base.py\u001b[0m in \u001b[0;36mwrapper\u001b[0;34m(estimator, *args, **kwargs)\u001b[0m\n\u001b[1;32m   1387\u001b[0m                 )\n\u001b[1;32m   1388\u001b[0m             ):\n\u001b[0;32m-> 1389\u001b[0;31m                 \u001b[0;32mreturn\u001b[0m \u001b[0mfit_method\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mestimator\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1390\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1391\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mwrapper\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_search.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, **params)\u001b[0m\n\u001b[1;32m   1022\u001b[0m                 \u001b[0;32mreturn\u001b[0m \u001b[0mresults\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1023\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1024\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_run_search\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mevaluate_candidates\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1025\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1026\u001b[0m             \u001b[0;31m# multimetric is determined here because in the case of a callable\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_search.py\u001b[0m in \u001b[0;36m_run_search\u001b[0;34m(self, evaluate_candidates)\u001b[0m\n\u001b[1;32m   1569\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_run_search\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mevaluate_candidates\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1570\u001b[0m         \u001b[0;34m\"\"\"Search all candidates in param_grid\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1571\u001b[0;31m         \u001b[0mevaluate_candidates\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mParameterGrid\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparam_grid\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1572\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1573\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_search.py\u001b[0m in \u001b[0;36mevaluate_candidates\u001b[0;34m(candidate_params, cv, more_results)\u001b[0m\n\u001b[1;32m    999\u001b[0m                     )\n\u001b[1;32m   1000\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1001\u001b[0;31m                 \u001b[0m_warn_or_raise_about_fit_failures\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mout\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0merror_score\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1002\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1003\u001b[0m                 \u001b[0;31m# For callable self.scoring, the return type is only know after\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_validation.py\u001b[0m in \u001b[0;36m_warn_or_raise_about_fit_failures\u001b[0;34m(results, error_score)\u001b[0m\n\u001b[1;32m    515\u001b[0m                 \u001b[0;34mf\"Below are more details about the failures:\\n{fit_errors_summary}\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    516\u001b[0m             )\n\u001b[0;32m--> 517\u001b[0;31m             \u001b[0;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mall_fits_failed_message\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    518\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    519\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mValueError\u001b[0m: \nAll the 5 fits failed.\nIt is very likely that your model is misconfigured.\nYou can try to debug the error by setting error_score='raise'.\n\nBelow are more details about the failures:\n--------------------------------------------------------------------------------\n5 fits failed with the following error:\nTraceback (most recent call last):\n  File \"/usr/local/lib/python3.12/dist-packages/sklearn/model_selection/_validation.py\", line 866, in _fit_and_score\n    estimator.fit(X_train, y_train, **fit_params)\n  File \"/usr/local/lib/python3.12/dist-packages/sklearn/base.py\", line 1382, in wrapper\n    estimator._validate_params()\n  File \"/usr/local/lib/python3.12/dist-packages/sklearn/base.py\", line 436, in _validate_params\n    validate_parameter_constraints(\n  File \"/usr/local/lib/python3.12/dist-packages/sklearn/utils/_param_validation.py\", line 98, in validate_parameter_constraints\n    raise InvalidParameterError(\nsklearn.utils._param_validation.InvalidParameterError: The 'n_neighbors' parameter of KNeighborsClassifier must be an int in the range [1, inf) or None. Got array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,\n       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]) instead.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "whtpdNdT8cLC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "77Sk-1Jv8cNy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "pQS3RW0F8cQa"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "URGtomBX8cSp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "rRBgxUkm8cW-"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
