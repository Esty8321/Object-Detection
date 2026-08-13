const input = document.querySelector('#imageInput');
const chooseButton = document.querySelector('#chooseButton');
const analyzeButton = document.querySelector('#analyzeButton');
const removeButton = document.querySelector('#removeButton');
const filePreview = document.querySelector('#filePreview');
const previewImage = document.querySelector('#previewImage');
const fileName = document.querySelector('#fileName');
const fileSize = document.querySelector('#fileSize');
const dropZone = document.querySelector('#dropZone');
const loading = document.querySelector('#loading');
const errorBox = document.querySelector('#error');
const results = document.querySelector('#results');

chooseButton.addEventListener('click', () => input.click());
removeButton.addEventListener('click', clearFile);
input.addEventListener('change', () => setFile(input.files[0]));
analyzeButton.addEventListener('click', analyze);

['dragenter', 'dragover'].forEach(name => dropZone.addEventListener(name, event => {
    event.preventDefault(); dropZone.classList.add('dragging');
}));
['dragleave', 'drop'].forEach(name => dropZone.addEventListener(name, event => {
    event.preventDefault(); dropZone.classList.remove('dragging');
}));
dropZone.addEventListener('drop', event => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    const transfer = new DataTransfer(); transfer.items.add(file); input.files = transfer.files; setFile(file);
});

function setFile(file) {
    if (!file) return clearFile();
    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    previewImage.src = URL.createObjectURL(file);
    filePreview.hidden = false;
    analyzeButton.disabled = false;
    errorBox.hidden = true;
}

function clearFile() {
    input.value = ''; filePreview.hidden = true; analyzeButton.disabled = true;
    previewImage.removeAttribute('src');
}

async function analyze() {
    const file = input.files[0];
    if (!file) return;
    loading.hidden = false; results.hidden = true; errorBox.hidden = true;
    analyzeButton.disabled = true;
    const form = new FormData(); form.append('image', file);
    try {
        const response = await fetch('/api/analyze', { method: 'POST', body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Analysis failed.');
        renderResults(data);
    } catch (error) {
        errorBox.textContent = error.message; errorBox.hidden = false;
    } finally {
        loading.hidden = true; analyzeButton.disabled = false;
    }
}

function renderResults(data) {
    document.querySelector('#peopleCount').textContent = `${data.people_count} ${data.people_count === 1 ? 'person' : 'people'} detected`;
    document.querySelector('#annotatedImage').src = data.annotated_image_url;
    const host = document.querySelector('#people'); host.innerHTML = '';
    if (!data.people.length) {
        host.innerHTML = '<div class="empty">No person was detected in this image.</div>';
    }
    data.people.forEach(person => {
        const available = person.regions.filter(region => region.status === 'available');
        const unavailable = person.regions.filter(region => region.status !== 'available');
        const card = document.createElement('article'); card.className = 'person-card';
        card.innerHTML = `
      <div class="person-header">
        <div><span class="person-number">${person.number}</span><div><h3>Person ${person.number}</h3><p>${available.length} of ${person.regions.length} regions available</p></div></div>
        <div class="metrics">
          ${metric('Gender', person.gender)}${metric('Visibility', person.visibility)}${metric('Occlusion', person.occlusion)}${metric('Quality', person.quality)}${metric('Rotation', `${person.rotation}°`)}
        </div>
      </div>
      <div class="region-grid">${available.map(regionCard).join('')}</div>
      ${unavailable.length ? `<details><summary>${unavailable.length} unavailable regions</summary><div class="unavailable-list">${unavailable.map(r => `<span><b>${label(r.name)}</b>${humanReason(r.reason)}</span>`).join('')}</div></details>` : ''}`;
        host.appendChild(card);
    });
    results.hidden = false;
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

const label = value => value.split('_').map(word => word[0].toUpperCase() + word.slice(1)).join(' ');
const metric = (name, value) => `<span><small>${name}</small><b>${label(String(value))}</b></span>`;
const regionCard = region => `<figure class="region-card"><img src="${region.image_url}" alt="${label(region.name)} crop"><figcaption><strong>${label(region.name)}</strong><span class="quality ${region.quality}">${label(region.quality)}</span></figcaption></figure>`;
const humanReason = reason => ({ need_visible_shoulder_and_hip: 'Shoulder and hip points were not reliable.', need_visible_elbow_and_wrist: 'Elbow and wrist points were not reliable.', need_visible_hip_and_knee: 'Hip and knee points were not reliable.', need_visible_knee_and_ankle: 'Knee and ankle points were not reliable.' }[reason] || label(reason || 'Not visible'));