/** Same-origin JSON client with consistent sanitized error messages. */
export async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let data;
  try { data = await response.json(); } catch { data = null; }
  if (!response.ok) throw new Error(data?.message || `${response.status} ${response.statusText}`);
  return data;
}

export const post = (path, body = {}) => api(path, { method: "POST", body: JSON.stringify(body) });

