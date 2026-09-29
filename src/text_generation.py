"""
Generate the text for the training of the model. The text is generated based on the information of the adverse reaction, the active principle, the medication, the frequency and the system affected.
"""


def generate_text(
    principio_activo: str,
    medicamento: str,
    reaccion_adversa: str,
    frecuencia: str,
    sistema: str,
) -> list[str]:

    text: list[str] = [
        f"El principio activo {principio_activo} del medicamento {medicamento} causa {reaccion_adversa}",
        f"{reaccion_adversa} es una reacción adversa de {principio_activo}",
        f"Reacción adversa: {reaccion_adversa} con frecuencia {frecuencia} en el {sistema}",
        f"Efecto secundario de {principio_activo}: {reaccion_adversa}",
        f"Síntoma reportado: {reaccion_adversa}",
        f"El tratamiento con {principio_activo} puede producir {reaccion_adversa} en el {sistema} con una frecuencia de {frecuencia}.",
        f"El medicamento {medicamento} con principio activo {principio_activo} puede causar {reaccion_adversa} de {frecuencia} en el {sistema}.",
    ]
    return text
