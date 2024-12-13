import React, { useState } from 'react';
import './App.css';

function PasswordChecker() {
  const [password, setPassword] = useState('');
  const [strength, setStrength] = useState('');
  const [loading, setLoading] = useState(false);

  const checkPasswordStrength = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/check-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ password }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Network response was not ok');
      }

      const data = await response.json();
      setStrength(data.strength);
    } catch (error) {
      console.error('Error:', error);
      setStrength('Error checking password strength.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Password Strength Checker</h1>
      <form onSubmit={checkPasswordStrength} className="p-4 bg-light rounded shadow">
        <label htmlFor="password">Enter your password:</label>
        <input
          type="password"
          id="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="form-control mb-3"
          required
        />
        <button type="submit" className="btn btn-success" disabled={loading}>
          {loading ? 'Checking...' : 'Check Strength'}
        </button>
      </form>
      {strength && <h2 className="mt-3">{`Password Strength: ${strength}`}</h2>}
    </div>
  );
}

export default PasswordChecker;
