const form = document.getElementById("loanForm")

form.addEventListener("submit", async function (e) {

    e.preventDefault()

    const data = {

        no_of_dependents: parseInt(document.getElementById("dependents").value),

        education: parseInt(document.getElementById("education").value),

        self_employed: parseInt(document.getElementById("self_employed").value),

        income_annum: parseFloat(document.getElementById("income_annum").value),

        loan_amount: parseFloat(document.getElementById("loan_amount").value),

        loan_term: parseInt(document.getElementById("loan_term").value),

        cibil_score: parseInt(document.getElementById("cibil_score").value),

        residential_assets_value: parseFloat(document.getElementById("residential_assets_value").value),

        commercial_assets_value: parseFloat(document.getElementById("commercial_assets_value").value),

        luxury_assets_value: parseFloat(document.getElementById("luxury_assets_value").value),

        bank_asset_value: parseFloat(document.getElementById("bank_asset_value").value)

    }

    try {

        const response = await fetch("http://127.0.0.1:8000/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)

        })

        const result = await response.json()

        localStorage.setItem("predictionResult", JSON.stringify(result))

        window.location.href = "result.html"

    } catch (error) {

        alert("Error connecting to backend")

    }

})