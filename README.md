# 🚀 PXR Model Deployment - Quick Reference Sheet

### 1. Logowanie na serwer (AWS)
```bash
ssh -i "mlops-key.pem" ubuntu@<TWÓJ_PUBLICZNY_ADRES_IP>
```

---

### 2. Zarządzanie infrastrukturą i dyskiem
*Jeśli kiedykolwiek zabraknie miejsca (standardowo 8GB to za mało dla Dockera i PyTorcha):*

*   **Sprawdzenie wolnego miejsca:** `df -h`
*   **Powiększenie partycji (dla NVMe):**
    ```bash
    sudo growpart /dev/nvme0n1 1
    sudo resize2fs /dev/nvme0n1p1
    ```
*   **Czyszczenie śmieci Dockera:** `docker system prune -f`

---

### 3. Budowanie i Konteneryzacja (BentoML)
*Pamiętaj o pliku `__init__.py` w folderze z modelem oraz poprawnej sekcji `include` w `bentofile.yaml`.*

*   **Budowanie Bento:**
    ```bash
    python3 -m bentoml build
    ```
*   **Tworzenie obrazu Docker:**
    ```bash
    # Użyj tagu wygenerowanego w poprzednim kroku
    python3 -m bentoml containerize pxr_service:<TAG>
    ```

---

### 4. Uruchamianie Serwisu
*   **Uruchomienie kontenera w tle:**
    ```bash
    docker run -d -p 3000:3000 pxr_service:<TAG>
    ```
*   **Podgląd logów na żywo (jeśli kontener nie wstaje):**
    ```bash
    docker logs -f <ID_KONTENERA>
    
    ```

---

### 5. Testowanie API
*   **Lokalnie z serwera (Curl):**
    ```bash
    curl -X POST http://localhost:3000/predict \
         -H "Content-Type: application/json" \
         -d '{"smiles": "CCO"}'
    ```
*   **Z Twojego komputera:**
    Uruchom skrypt `test_aws.py`, upewniając się, że adres IP jest aktualny, a port **3000** jest otwarty w **AWS Security Groups**.

---

### 6. Porządki po pracy
*   **Zatrzymanie kontenera:** `docker stop <ID_KONTENERA>`
*   **Wyjście z serwera:** `exit`
*   **WAŻNE:** Zatrzymaj instancję w panelu AWS (**Instance State -> Stop**), aby uniknąć kosztów.

### 7. Odpalenie Bento Service (bez Dockera)
Jeśli chcesz przetestować usługę bezpośrednio w terminalu (np. przed budowaniem obrazu):

    bentoml serve service:PXRService