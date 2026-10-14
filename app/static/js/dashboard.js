const searchInput = document.getElementById("machineSearch");
const table = document.getElementById("machineTable");

if (searchInput && table) {

    searchInput.addEventListener("keyup", function () {

        const searchValue =
            searchInput.value.toLowerCase();

        const rows =
            table.querySelectorAll("tbody tr");

        rows.forEach(function (row) {

            const text =
                row.textContent.toLowerCase();

            if (text.includes(searchValue)) {

                row.style.display = "";

            } else {

                row.style.display = "none";

            }

        });

    });

}