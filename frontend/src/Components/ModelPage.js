import InputForm from "./InputForm"
import Grid from '@mui/material/Grid';
import { useState } from "react";
import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormHelperText from '@mui/material/FormHelperText';
import FormControl from '@mui/material/FormControl';
import Select from '@mui/material/Select';
import FlightImage from "./flight.jpg"

const ModelPage = () => {

    const domains = ["Flight"]
    const background_images = {
        Flight: FlightImage
    }

    const [domain, setDomain] = useState(domains[0]);

    const handleChange = (event) => {
        setDomain(event.target.value);
    };


    return (
        <div style={{
            backgroundImage: `url(${background_images[domain]})`,
            backgroundPosition: 'center',
            backgroundRepeat: "no-repeat",
            backgroundSize: 'cover'
        }}>
            <FormControl sx={{ m: 1, minWidth: 120 }}>
                <InputLabel id="ddomain-select">Domain</InputLabel>
                <Select
                    labelId="domain-select-helper-label"
                    id="domain-select-helper"
                    value={domain}
                    label="Domain"
                    onChange={handleChange}
                >
                    {domains.map(domain => <MenuItem key={domain} value={domain}>{domain}</MenuItem>)}
                </Select>
                <FormHelperText>Select your domain</FormHelperText>
            </FormControl>


            <Grid container rowSpacing={5}>
                <Grid item xs={12} sm={12} md={12} lg={2.5} xl={2}></Grid>
                <Grid item xs={12} sm={12} md={12} lg={7} xl={8}>
                    <InputForm />
                </Grid>
                <Grid item xs={12} sm={12} md={12} lg={2.5} xl={2}></Grid>
            </Grid>
        </div>
    )
}

export default ModelPage