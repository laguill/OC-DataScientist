# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    import shap
    import matplotlib.pyplot as plt
    return (
        LogisticRegression,
        SentenceTransformer,
        accuracy_score,
        classification_report,
        np,
        pd,
        plt,
        shap,
        train_test_split,
    )


@app.cell
def _(pd):
    # ============================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ============================================================================

    # Sample data (replace with your actual DataFrame)
    sample_data = {
        'preprocessed_description': [
            'soft baby blanket organic cotton',
            'baby bottle sterilizer electric',
            'natural face cream anti aging',
            'lipstick matte finish long lasting',
            'laptop gaming high performance',
            'wireless mouse ergonomic design',
            'wall clock modern minimalist',
            'decorative cushion covers festive',
            'bedsheet cotton king size',
            'curtains blackout bedroom',
            'non stick frying pan',
            'dinner plates ceramic set',
            'luxury watch chronograph men',
            'smartwatch fitness tracker women'
        ] * 5,  # Repeat for more data
        'main_category': [
            'baby care', 'baby care',
            'beauty and personal care', 'beauty and personal care',
            'computers', 'computers',
            'home decor & festive needs', 'home decor & festive needs',
            'home furnishing', 'home furnishing',
            'kitchen & dining', 'kitchen & dining',
            'watches', 'watches'
        ] * 5
    }

    df = pd.DataFrame(sample_data)

    # For your actual data, use:
    # df = pd.read_csv('your_data.csv')

    print(f"Dataset shape: {df.shape}")
    print(f"\nClass distribution:\n{df['main_category'].value_counts()}\n")


    return (df,)


@app.cell(column=1)
def _():
    # Dropdown modèles
    st_models = {
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
        "MPNet": "sentence-transformers/all-mpnet-base-v2",
        "BERT Uncased": "bert-base-uncased",
    }
    return


@app.cell
def _(SentenceTransformer, df):
    # ============================================================================
    # STEP 2: LOAD SENTENCE TRANSFORMER & CREATE EMBEDDINGS
    # ============================================================================

    print("Loading Sentence Transformer (all-MiniLM-L6-v2)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Creating sentence embeddings...")
    X_embeddings = embedding_model.encode(
        df['preprocessed_description'].tolist(),
        show_progress_bar=True
    )
    y = df['main_category'].values

    print(f"Embedding shape: {X_embeddings.shape}")
    return X_embeddings, embedding_model, y


@app.cell
def _(
    LogisticRegression,
    X_embeddings,
    accuracy_score,
    classification_report,
    df,
    train_test_split,
    y,
):
    # ============================================================================
    # STEP 3: TRAIN LOGISTIC REGRESSION MODEL
    # ============================================================================

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_embeddings, y, df.index, test_size=0.3, random_state=42, stratify=y
    )

    print("\nTraining Logistic Regression model...")
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return (clf,)


@app.cell
def _(clf, df, embedding_model, np, shap):
    # ============================================================================
    # STEP 4: WORD-LEVEL SHAP EXPLANATION FUNCTION
    # ============================================================================

    def explain_sentence_words(text, embedding_model, clf, n_samples=100, verbose=True):
        """
        Explains which words in a sentence influence the prediction.
    
        Parameters:
        -----------
        text : str
            The sentence to explain
        embedding_model : SentenceTransformer
            The embedding model
        clf : LogisticRegression
            The trained classifier
        n_samples : int
            Number of samples for SHAP (higher = more accurate but slower)
        verbose : bool
            Whether to print progress
        
        Returns:
        --------
        shap_values : array
            SHAP values for each word and each class
        words : list
            List of words in the sentence
        prediction_info : dict
            Prediction details (class, probabilities)
        """
    
        words = text.split()
    
        if verbose:
            print(f"  Analyzing {len(words)} words...")
    
        def predict_from_word_presence(word_presence_matrix):
            """
            Predicts based on which words are present.
            word_presence_matrix: binary matrix where 1 = include word, 0 = exclude
            """
            predictions = []
        
            for presence_vector in word_presence_matrix:
                # Build sentence from present words
                active_words = [words[i] for i in range(len(words)) 
                              if presence_vector[i] == 1]
            
                # If no words present, use full sentence (fallback)
                if len(active_words) == 0:
                    sentence = text
                else:
                    sentence = " ".join(active_words)
            
                # Get embedding and predict
                embedding = embedding_model.encode([sentence])
                pred_probs = clf.predict_proba(embedding)[0]
                predictions.append(pred_probs)
        
            return np.array(predictions)
    
        # Create SHAP explainer with baseline (all words present)
        explainer = shap.KernelExplainer(
            predict_from_word_presence,
            np.ones((1, len(words)))
        )
    
        # Calculate SHAP values
        if verbose:
            print(f"  Computing SHAP values (n_samples={n_samples})...")
    
        shap_values = explainer.shap_values(
            np.ones((1, len(words))),
            nsamples=n_samples
        )
    
        # Get prediction info
        embedding = embedding_model.encode([text])
        pred_probs = clf.predict_proba(embedding)[0]
        predicted_class_idx = np.argmax(pred_probs)
        predicted_class = clf.classes_[predicted_class_idx]
    
        prediction_info = {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'probabilities': pred_probs,
            'all_classes': clf.classes_
        }
    
        return shap_values, words, prediction_info

    # ============================================================================
    # STEP 6: GLOBAL EXPLANATIONS - Overall Word Patterns
    # ============================================================================

    print("\n" + "="*70)
    print("GLOBAL EXPLANATIONS: Word Importance Patterns Across Dataset")
    print("="*70)

    # Collect word importance across multiple samples per class
    n_samples_per_class = 3
    word_importance_global = {class_name: {} for class_name in clf.classes_}

    print("\nAnalyzing samples from each category...")

    for class_name in clf.classes_:
        print(f"\n  Processing: {class_name}")
    
        # Get unique samples from this class
        class_samples = df[df['main_category'] == class_name]['preprocessed_description'].unique()
        class_samples = class_samples[:n_samples_per_class]
    
        for text in class_samples:
            # Get word-level SHAP values
            shap_values, words, pred_info = explain_sentence_words(
                text, embedding_model, clf, n_samples=50, verbose=False
            )
        
            # Get index for this class
            class_idx = list(clf.classes_).index(class_name)
        
            # Store word importance
            for word, shap_val in zip(words, shap_values[class_idx][0]):
                if word not in word_importance_global[class_name]:
                    word_importance_global[class_name][word] = []
                word_importance_global[class_name][word].append(shap_val)

    # Display global word importance
    print("\n" + "="*70)
    print("🌍 GLOBAL WORD IMPORTANCE BY CATEGORY")
    print("="*70)

    for class_name in clf.classes_:
        print(f"\n{'='*70}")
        print(f"📂 CATEGORY: {class_name.upper()}")
        print(f"{'='*70}")
    
        # Average SHAP values per word
        word_avg_importance = {}
        for word, shap_list in word_importance_global[class_name].items():
            word_avg_importance[word] = {
                'mean': np.mean(shap_list),
                'std': np.std(shap_list),
                'count': len(shap_list)
            }
    
        # Sort by absolute mean importance
        sorted_words = sorted(
            word_avg_importance.items(),
            key=lambda x: abs(x[1]['mean']),
            reverse=True
        )[:10]
    
        print(f"\nTop 10 Most Important Words:\n")
        for rank, (word, stats) in enumerate(sorted_words, 1):
            mean_shap = stats['mean']
            impact = "STRONG +" if mean_shap > 0.01 else "STRONG -" if mean_shap < -0.01 else "MODERATE"
            print(f"  {rank:2d}. {word:20s} → {mean_shap:+.4f}  [{impact}]")


    return (explain_sentence_words,)


@app.cell
def _(clf, df, embedding_model, explain_sentence_words, np, plt, shap):
    # ============================================================================
    # STEP 5: LOCAL EXPLANATIONS - Individual Sentence Analysis
    # ============================================================================

    print("\n" + "="*70)
    print("LOCAL EXPLANATIONS: Word Importance for Individual Predictions")
    print("="*70)

    # Select examples to analyze (one from each category)
    example_texts = []
    for category in df['main_category'].unique():
        example = df[df['main_category'] == category].iloc[0]['preprocessed_description']
        example_texts.append((example, category))

    # Analyze each example
    for i, (text, true_label) in enumerate(example_texts[:5]):  # Analyze first 5
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*70}")
        print(f"Text: '{text}'")
        print(f"True Label: {true_label}\n")
    
        # Get SHAP explanation
        shap_values, words, pred_info = explain_sentence_words(
            text, embedding_model, clf, n_samples=100
        )
    
        predicted_class = pred_info['predicted_class']
        predicted_class_idx = pred_info['predicted_class_idx']
        confidence = pred_info['probabilities'][predicted_class_idx]
    
        print(f"\n→ Predicted: {predicted_class}")
        print(f"→ Confidence: {confidence:.3f}")
    
        # Display word importance for predicted class
        print(f"\n📊 Word Importance for '{predicted_class}':\n")
    
        word_shap_pairs = list(zip(words, shap_values[predicted_class_idx][0]))
        word_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    
        for word, shap_val in word_shap_pairs:
            # Visual representation
            if shap_val > 0:
                direction = "→ INCREASES"
                bar = "█" * min(int(abs(shap_val) * 100), 30)
            else:
                direction = "← DECREASES"
                bar = "▓" * min(int(abs(shap_val) * 100), 30)
        
            print(f"   {word:18s} {shap_val:+.4f}  {direction:15s} {bar}")
    
        # Create visualization
        plt.figure(figsize=(10, 6))
    
        shap.plots.bar(
            shap.Explanation(
                values=shap_values[predicted_class_idx][0],
                feature_names=words
            ),
            show=False
        )
    
        plt.title(f"Word Importance: '{text}'\n→ {predicted_class} (confidence: {confidence:.2f})", 
                  fontsize=12, pad=20)
        plt.xlabel("SHAP Value (Impact on Prediction)", fontsize=10)
        plt.tight_layout()
    
        filename = f'shap_local_example_{i+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved visualization: '{filename}'")
        plt.close()
    
        # Show top 3 classes with probabilities
        print(f"\n📈 Top 3 Class Probabilities:")
        top_3_idx = np.argsort(pred_info['probabilities'])[-3:][::-1]
        for idx in top_3_idx:
            class_name = pred_info['all_classes'][idx]
            prob = pred_info['probabilities'][idx]
            print(f"   {class_name:30s} {prob:.3f}")


    return (example_texts,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md("""### SHAP Feature Importance for Selected Model""")
    return


@app.cell
def _(example_texts):
    # ============================================================================
    # STEP 7: SUMMARY & INTERPRETATION GUIDE
    # ============================================================================

    print("\n" + "="*70)
    print("📚 INTERPRETATION GUIDE")
    print("="*70)

    print("""
    LOCAL EXPLANATIONS (Individual Predictions):
    ---------------------------------------------
    → Shows which words influenced a SPECIFIC prediction
    → Positive SHAP values: word pushes toward the predicted class
    → Negative SHAP values: word pushes away from the predicted class
    → Larger absolute value = stronger influence

    Example: For "baby bottle sterilizer" → "baby care"
      - "baby" has high positive value → strongly indicates baby care
      - "sterilizer" has moderate positive value → supports the prediction
      - "bottle" might have lower value → less discriminative

    GLOBAL EXPLANATIONS (Overall Patterns):
    ----------------------------------------
    → Shows which words are GENERALLY important across many examples
    → Identifies discriminative vocabulary for each category
    → Helps understand what the model learned

    Example findings:
      - Baby care: "baby", "infant", "newborn" are key indicators
      - Computers: "laptop", "gaming", "processor" drive predictions
      - Beauty: "cream", "lipstick", "skincare" are important

    ACTIONABLE INSIGHTS:
    -------------------
    ✅ Use local explanations to debug individual misclassifications
    ✅ Use global explanations to understand model behavior overall
    ✅ Look for unexpected words with high importance (potential issues)
    ✅ Verify important words align with domain knowledge
    """)

    print("\n✅ Analysis complete! Check the generated PNG files for visualizations.")
    print(f"   Generated {len(example_texts[:5])} local explanation plots.\n")
    return


if __name__ == "__main__":
    app.run()
