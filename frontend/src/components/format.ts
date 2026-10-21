export function formatModelName(name: string) {
  return name === 'xgboost' ? 'XGBoost' : name.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}
