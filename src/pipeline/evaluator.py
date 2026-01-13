import numpy as np

class Evaluator:
    def __init__(self):
        pass
    
    def calculate_mrr(y_true_ids, y_pred_top10):
        """
        y_true_ids: List with the real IDs (from y_train or y_val)
        y_pred_top10: List of lists, each one with the 10 recommended IDs
        """
        rr_scores = []
        
        for true_id, pred_list in zip(y_true_ids, y_pred_top10):
            if true_id in pred_list:
                # El ranking es el índice + 1 (porque el índice empieza en 0)
                rank = pred_list.index(true_id) + 1
                rr_scores.append(1.0 / rank)
            else:
                # Si el job real no está en tu Top 10, sumas 0 
                rr_scores.append(0.0)
                
        return np.mean(rr_scores)
    
    def calculate_action_accuracy(y_true_actions, y_pred_actions):
        """
        y_true_actions: list of real actions (0 o 1)
        y_pred_actions: list of predicted actions by your model using the threshold theta
        """
        correctas = sum(1 for true, pred in zip(y_true_actions, y_pred_actions) if true == pred)
        accuracy = correctas / len(y_true_actions)
        
        return accuracy

    def calculate_final_score(mrr_score, action_accuracy):
        """
        mrr_score: MRR score
        action_accuracy: Accuracy score
        """
        return (0.7 * mrr_score) + (0.3 * action_accuracy)
