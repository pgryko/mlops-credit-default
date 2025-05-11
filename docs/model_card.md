# Credit Card Default Prediction Model Card

## Model Overview
**Model Name**: Credit Default Classifier
**Version**: 2.0
**Type**: CatBoost Classifier
**Task**: Binary classification for credit card default prediction
**Last Updated**: May 2025
**Model ID**: catboost_model_credit_default
**Repository**: https://github.com/pgryko/mlops-credit-default

## Intended Use
- **Primary Use**: Predict the probability of credit card default for risk assessment
- **Intended Users**: Financial institutions, credit risk analysts, loan officers
- **Use Cases**:
  - Portfolio risk assessment
  - Credit approval decision support
  - Default probability estimation for existing customers
  - Risk-based pricing models
- **Out-of-Scope Uses**:
  - Individual credit scoring without human oversight
  - Real-time transaction fraud detection
  - Loan approval automation without additional checks
  - Predictive collections or harassment
  - Credit scoring in regions with different economic conditions than Taiwan

## Training Data
- **Source**: UCI ML Credit Card Default Dataset
- **Size**: 30,000 records
- **Time Period**: April-September 2005
- **Demographics**: Taiwan credit card holders
- **Features**:
  - Demographic: Age, Sex, Education, Marriage
  - Credit: Limit balance (continuous)
  - Payment History: Last 6 months payment status (categorical, -2 to 8)
  - Bill Amounts: Last 6 months bill amounts (continuous)
  - Payment Amounts: Last 6 months payment amounts (continuous)
- **Target**: Binary default payment classification (0: Non-default, 1: Default)
- **Class Distribution**: Imbalanced (78% non-default, 22% default)

## Model Architecture
- **Framework**: CatBoost v1.2.3
- **Type**: Gradient Boosting Decision Trees
- **Key Parameters**:
  - Learning Rate: 0.03
  - Tree Depth: 9
  - L2 Regularization: 0.8
  - Iterations: 760
  - Bagging Temperature: 1.2
  - Random Strength: 0.5
  - Class Weights: Balanced using data distribution (Default: 3.54, Non-default: 0.28)
- **Features Handling**:
  - Categorical features processed with CatBoost's ordered boosting
  - Missing values imputed during preprocessing
  - No feature scaling required (tree-based model)
  - One-hot encoding for Education and Marriage features

## Performance Metrics
### Overall Performance
- Precision-Recall AUC: 0.63
- F1 Score: 0.477
- Accuracy: 0.792
- Balanced Accuracy: 0.681
- Default Detection Rate: 58.4%
- False Default Rate: 10.5%

### Business Metrics
- Approval Rate: 81.7%
- Default Rate Among Approved: 16.2%
- Average Cost Per Decision: $230.14
- Total Cost (per 1000 customers): $230,140
- Optimal Threshold: 0.38 (cost-optimized)

### SHAP Feature Importance
- Demographics: 0.0189
- Payment History: 0.0299
- Bill Amounts: 0.0313
- Payment Amounts: 0.0283

### Performance Across Groups
- Gender Gap (M/F Approval Diff): 3.2%
- Education Level Disparity: 4.7%
- Age Group Variance: 5.1%
- Income Level Proxy Variance: 7.3%

## Limitations and Biases

### Known Limitations
1. **Geographic Limitation**:
   - Training data is from Taiwan only
   - May not generalize well to other regions with different economic conditions
   - Cultural differences in credit usage not accounted for

2. **Temporal Limitation**:
   - Based on 2005 data
   - Economic conditions and consumer behavior have changed significantly
   - Does not account for macroeconomic factors like recessions

3. **Feature Limitations**:
   - No income information (crucial for ability-to-pay assessment)
   - Limited to credit card history (no mortgage or other loan types)
   - No external credit bureau data (credit score, total debt)
   - Missing behavioral features (spending categories, transaction frequency)

### Potential Biases
1. **Demographic Biases**:
   - Education level representation may vary from general population
   - Age distribution skews toward middle-aged customers (25-45)
   - Gender-based approval rate differences (3.2% higher for males)
   - No ethnicity data to assess racial bias (critical limitation)

2. **Economic Biases**:
   - Credit limit variations across demographic groups (proxy for income bias)
   - Payment patterns affected by income level
   - Regional economic conditions impact default rates differently
   - Penalty for missed payments may disproportionately affect lower-income groups

### Mitigations
1. **Data Validation**:
   - Comprehensive range checks for all numeric features
   - Missing value handling with domain-appropriate imputation
   - Outlier detection and treatment with winsorization
   - Stratified sampling to preserve class distribution

2. **Model Design**:
   - Class weight balancing to address imbalanced dataset
   - Cost-sensitive optimization with business-specific costs
   - Regular performance monitoring across demographic groups
   - Feature importance analysis to identify potential bias sources

3. **Deployment Safeguards**:
   - Probability threshold optimization using cost matrix
   - SHAP-based prediction explanations for transparency
   - Drift detection mechanisms with statistical tests
   - Monthly performance monitoring with detailed breakdowns

## Ethical Considerations
1. **Fairness**:
   - Monitor approval rates across demographic groups monthly
   - Regular bias audits with standardized metrics
   - Transparent decision explanations for all rejected applications
   - Bias remediation process with cross-functional review

2. **Privacy**:
   - No personally identifiable information used in model
   - Aggregate statistics only for reporting
   - Secure data handling with encryption
   - Data retention policies aligned with regulations

3. **Transparency**:
   - SHAP explanations for all predictions
   - Clear documentation of limitations for end users
   - Regular performance reporting to stakeholders
   - Customer-facing explanation system

4. **Regulatory Compliance**:
   - Adheres to ECOA (Equal Credit Opportunity Act) guidelines
   - Compliant with FCRA (Fair Credit Reporting Act) requirements
   - Documentation sufficient for regulatory review
   - Regular compliance assessments

## Model Updates and Maintenance
- **Update Frequency**: Quarterly retraining
- **Monitoring**:
  - Data drift detection (PSI, KS tests)
  - Performance metrics tracking (F1, PR-AUC)
  - Business impact assessment (cost per decision)
  - Group fairness metrics tracking
- **Versioning**:
  - MLflow model registry with champion/challenger approach
  - Git version control for all code and configuration
  - Documentation updates with each release
  - Automated testing of critical components

## Additional Information
- **Contact**: Piotr Gryko
- **Citation**: Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
- **Original Dataset**: I-Cheng Yeh and Che-hui Lien, "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients", Expert Systems with Applications, 2009.
- **License**: MIT
- **Documentation**: See project README.md and notebooks directory

## Model Governance
- **Approval Process**:
  1. Technical review by ML engineering team
  2. Business metrics validation by analytics team
  3. Bias assessment by governance committee
  4. Production deployment approval by leadership
- **Review Schedule**: Quarterly
- **Incident Response**:
  1. Automated alerts for performance degradation
  2. On-call engineer notification
  3. Impact assessment and mitigation
  4. Root cause analysis and documentation
- **Compliance**:
  - FCRA (Fair Credit Reporting Act)
  - ECOA (Equal Credit Opportunity Act)
  - GDPR (where applicable)
  - Model Risk Management guidelines

## Feedback and Updates
We actively monitor model performance and welcome feedback. Please report any issues or concerns through:
- GitHub Issues: https://github.com/pgryko/mlops-credit-default/issues
- Project Contact: Piotr Gryko

This model card should be reviewed and updated with each model iteration or when significant changes occur in the model's performance or usage patterns.