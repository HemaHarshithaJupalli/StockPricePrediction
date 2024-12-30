  document.addEventListener("DOMContentLoaded", function() {
    const faqItems = document.querySelectorAll('.list input[type="radio"]');
    faqItems.forEach(function(item) {
        item.addEventListener("change", function() {
            const con = this.parentNode.querySelector('.con');
            if (this.checked) {
                con.style.maxHeight = con.scrollHeight + "100px";
            } else {
                con.style.maxHeight = 0;
            }
        });
    });
});
const faqs = document.querySelectorAll(".faq");
faqs.forEach(faq =>{
    faq.addEventListener("click",()=>{
        faq.classList.toggle("active");
    });
});
