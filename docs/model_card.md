# Credit Card Default Prediction Model Card

## Model Overview
**Model Name**: Credit Default Classifier  
**Version**: 1.0  
**Type**: CatBoost Classifier  
**Task**: Binary classification for credit card default prediction  
**Last Updated**: April 2025

## Intended Use
- **Primary Use**: Predict the probability of credit card default for risk assessment
- **Intended Users**: Financial institutions, credit risk analysts
- **Out-of-Scope Uses**: 
  - Individual credit scoring without human oversight
  - Real-time transaction fraud detection
  - Loan approval automation without additional checks

## Training Data
- **Source**: UCI ML Credit Card Default Dataset
- **Size**: ~30,000 records
- **Time Period**: 2005
- **Demographics**: Taiwan credit card holders
- **Features**:
  - Demographic: Age, Sex, Education, Marriage
  - Credit: Limit balance
  - Payment History: Last 6 months payment status
  - Bill Amounts: Last 6 months bill amounts
  - Payment Amounts: Last 6 months payment amounts

## Model Architecture
- **Framework**: CatBoost
- **Type**: Gradient Boosting Decision Trees
- **Key Parameters**:
  - Learning Rate: Optimized via Optuna
  - Tree Depth: Optimized via Optuna
  - L2 Regularization: Optimized via Optuna
  - Class Weights: Balanced using data distribution

## Performance Metrics
### Overall Performance
- Precision-Recall AUC: X.XX
- F1 Score: X.XX
- Accuracy: X.XX

### Business Metrics
- Approval Rate: XX%
- Default Rate: XX%
- Average Cost per Decision: $XX.XX
- False Positive Cost (wrongly denied credit): $1.00
- False Negative Cost (default loss): $5.00

### Performance Across Groups
- Education Level Groups:
  - Graduate: F1 = X.XX
  - University: F1 = X.XX
  - High School: F1 = X.XX
  - Other: F1 = X.XX

- Age Groups:
  - 20-30: F1 = X.XX
  - 31-40: F1 = X.XX
  - 41-50: F1 = X.XX
  - 51+: F1 = X.XX

## Limitations and Biases

### Known Limitations
1. **Geographic Limitation**: 
   - Training data is from Taiwan only
   - May not generalize well to other regions

2. **Temporal Limitation**:
   - Based on 2005 data
   - Economic conditions and consumer behavior have changed

3. **Feature Limitations**:
   - No income information
   - Limited to credit card history
   - No external credit bureau data

### Potential Biases
1. **Demographic Biases**:
   - Education level representation may vary
   - Age distribution might not be uniform
   - Gender-based approval rate differences

2. **Economic Biases**:
   - Credit limit variations across groups
   - Payment patterns affected by income level
   - Regional economic conditions impact

### Mitigations
1. **Data Validation**:
   - Comprehensive range checks
   - Missing value handling
   - Outlier detection and treatment

2. **Model Design**:
   - Class weight balancing
   - Cost-sensitive optimization
   - Regular performance monitoring across groups

3. **Deployment Safeguards**:
   - Probability threshold optimization
   - Prediction explanations using SHAP
   - Drift detection mechanisms

## Ethical Considerations
1. **Fairness**:
   - Monitor approval rates across demographic groups
   - Regular bias audits
   - Transparent decision explanations

2. **Privacy**:
   - No personally identifiable information used
   - Aggregate statistics only
   - Secure data handling

3. **Transparency**:
   - SHAP explanations for predictions
   - Clear documentation of limitations
   - Regular performance reporting

## Model Updates and Maintenance
- **Update Frequency**: Quarterly retraining
- **Monitoring**:
  - Data drift detection
  - Performance metrics tracking
  - Business impact assessment
- **Versioning**:
  - MLflow model registry
  - Git version control
  - Documentation updates

## Additional Information
- **Contact**: [Team Contact]
- **Citation**: [Dataset Citation]
- **License**: MIT
- **Documentation**: [Link to Full Documentation]

## Model Governance
- **Approval Process**: [Description]
- **Review Schedule**: Quarterly
- **Incident Response**: [Protocol Reference]
- **Compliance**: [Relevant Standards]

## Feedback and Updates
We actively monitor model performance and welcome feedback. Please report any issues or concerns through:
- GitHub Issues
- Team Contact
- Regular Review Meetings

This model card should be reviewed and updated with each model iteration or when significant changes occur in the model's performance or usage patterns.