from utilsforecast.losses import mase, smape, rmae, mae

LOSSES = {
    'mase': mase,
    'smape': smape,
    'rmae': rmae,
    'mae': mae
}
